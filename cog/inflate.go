package cog

import (
	"encoding/binary"
	"errors"
	"math/bits"
	"sync"
)

// inflate decodes a whole raw Deflate stream (RFC 1951) in one call, into
// a buffer that holds the block's decoded size. It exists because a TIFF
// block is decoded whole, into a buffer whose size is known in advance,
// which a streaming decoder cannot take advantage of: compress/flate and
// klauspost/compress/flate decode into a 32 KiB window and copy out, read
// their input a byte at a time through an interface, and copy matches a
// byte at a time. Here matches are copied from the output itself, eight
// bytes at a time, and the input is read eight bytes at a time into a
// 64-bit bit buffer, as libdeflate does.
//
// It returns exactly what compress/flate would give a caller that reads
// up to len(dst) bytes (readInto): decoding stops at len(dst) bytes,
// whatever follows; a stream that ends early returns what it decoded
// with no error, for the caller's length check; and a corrupt stream is
// an error. FuzzInflate holds it to that, against compress/flate.
func inflate(dst, src []byte) ([]byte, error) {
	d := inflaters.Get().(*inflater)
	defer inflaters.Put(d)
	return d.inflate(dst, src)
}

var inflaters = sync.Pool{New: func() any { return new(inflater) }}

var errInflateCorrupt = errors.New("corrupt Deflate data")

// The decode tables are libdeflate's layout: a primary table indexed by
// the next tableBits bits of input, whose entries for longer codes point
// at a subtable indexed by the bits after those.
const (
	litlenBits   = 11
	distBits     = 8
	precodeBits  = 7
	litlenEnough = 2342 // the most entries 288 symbols of up to 15 bits can need with an 11-bit primary table
	distEnough   = 402  // the same for 32 symbols and an 8-bit table
)

// An entry is a uint32:
//
//	bits 0-4    how many bits it consumes: the code's length, or the
//	            primary table's bits for a subtable pointer
//	bits 8-11   its kind
//	bits 12-15  extra bits after the code (lengths, distances), or a
//	            subtable's index bits
//	bits 16-31  the value: a literal, a base length or distance, or a
//	            subtable's offset
const (
	kindLiteral = iota
	kindLength
	kindEnd
	kindDist
	kindSub
	kindInvalid
)

func hentry(kind, nbits, extra, value uint32) uint32 {
	return value<<16 | extra<<12 | kind<<8 | nbits
}

// lengthBase and lengthExtra are length symbols 257-285's base lengths
// and extra bits; distBase and distExtra the same for distance symbols
// 0-29.
var (
	lengthBase = [29]uint32{3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
		35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258}
	lengthExtra = [29]uint32{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
		3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0}
	distBase = [30]uint32{1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193,
		257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577}
	distExtra = [30]uint32{0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
		7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13}
	// precodeOrder is the order the code-length code's lengths come in.
	precodeOrder = [19]uint8{16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15}
)

// fixedLitlen and fixedDist are the tables of the fixed Huffman codes
// (RFC 1951 §3.2.6), built once.
var fixedTables = sync.OnceValue(func() *inflater {
	d := new(inflater)
	var lens [288 + 32]uint8
	for i := range 288 {
		switch {
		case i < 144:
			lens[i] = 8
		case i < 256:
			lens[i] = 9
		case i < 280:
			lens[i] = 7
		default:
			lens[i] = 8
		}
	}
	for i := range 32 {
		lens[288+i] = 5
	}
	if !d.buildLitlen(lens[:288]) || !d.buildDist(lens[288:]) {
		panic("cog: the fixed Huffman codes do not build")
	}
	return d
})

type inflater struct {
	litlen  [litlenEnough]uint32
	dist    [distEnough]uint32
	precode [1 << precodeBits]uint32
	lens    [286 + 30]uint8
	// litlenNeed and distNeed are the bits compress/flate insists on
	// having before it decodes a symbol: the code's shortest length, and
	// for literals and lengths at least the end-of-block code's. Only a
	// stream cut short notices: it ends a symbol earlier than the bits
	// alone would.
	litlenNeed, distNeed int
}

// shortest is the shortest non-zero length in lens, or 0.
func shortest(lens []uint8) int {
	m := 0
	for _, l := range lens {
		if l != 0 && (m == 0 || int(l) < m) {
			m = int(l)
		}
	}
	return m
}

// build fills table with the canonical Huffman code of lens, symbol s
// becoming entryOf(s): the primary table of tableBits bits, and subtables
// after it. It reports false for a code compress/flate rejects: one that
// oversubscribes its lengths, or leaves some unassigned but is not the
// single one-bit code zlib allows. An empty code is accepted, and any
// symbol decoded with it is corrupt, as in compress/flate.
func build(table []uint32, lens []uint8, tableBits uint, entryOf func(sym int, nbits uint32) uint32) bool {
	var count [16]int
	maxLen := 0
	for _, l := range lens {
		count[l]++
		maxLen = max(maxLen, int(l))
	}
	count[0] = 0
	invalid := hentry(kindInvalid, 0, 0, 0)
	primary := table[:1<<tableBits]
	if maxLen == 0 {
		for i := range primary {
			primary[i] = invalid
		}
		return true
	}
	// Completeness, as compress/flate's huffmanDecoder.init checks it.
	code := 0
	var next [16]int
	for l := 1; l <= maxLen; l++ {
		code <<= 1
		next[l] = code
		code += count[l]
	}
	if code != 1<<maxLen && (code != 1 || maxLen != 1) {
		return false
	}
	for i := range primary {
		primary[i] = invalid
	}
	// Codes are assigned in order of length, then symbol, and read from
	// the stream least significant bit first, so each is bit-reversed
	// to index the table.
	end := 1 << tableBits // where the next subtable goes
	for sym, l := range lens {
		if l == 0 {
			continue
		}
		c := next[l]
		next[l]++
		rev := int(bits.Reverse16(uint16(c)) >> (16 - uint(l))) // #nosec G115 -- c < 1<<15
		if uint(l) <= tableBits {
			e := entryOf(sym, uint32(l))
			for i := rev; i < len(primary); i += 1 << l {
				primary[i] = e
			}
			continue
		}
		// A long code: its first tableBits bits pick a subtable, sized
		// for the longest code sharing that prefix.
		prefix := rev & (1<<tableBits - 1)
		p := primary[prefix]
		if p>>8&0xf != kindSub {
			subBits := subtableBits(lens, sym, l, tableBits, prefix)
			if end+1<<subBits > len(table) {
				return false // cannot happen for a complete code
			}
			p = hentry(kindSub, uint32(tableBits), uint32(subBits), uint32(end)) // #nosec G115 -- small
			primary[prefix] = p
			for i := end; i < end+1<<subBits; i++ {
				table[i] = invalid
			}
			end += 1 << subBits
		}
		start, subBits := int(p>>16), uint(p>>12&0xf)
		sub := table[start : start+1<<subBits]
		e := entryOf(sym, uint32(uint(l)-tableBits)) // #nosec G115 -- l > tableBits, at most 15
		for i := rev >> tableBits; i < len(sub); i += 1 << (uint(l) - tableBits) {
			sub[i] = e
		}
	}
	return true
}

// subtableBits is the index width of the subtable for codes whose first
// tableBits bits (reversed) are prefix: the length of the longest such
// code, less tableBits. The codes after sym are the only ones that can
// share its prefix and be longer, so only they are looked at, with the
// canonical code recomputed for each.
func subtableBits(lens []uint8, sym int, l uint8, tableBits uint, prefix int) uint {
	// Recompute the canonical codes: small tables, built rarely.
	var count [16]int
	for _, x := range lens {
		count[x]++
	}
	count[0] = 0
	var next [16]int
	code := 0
	for i := 1; i < 16; i++ {
		code <<= 1
		next[i] = code
		code += count[i]
	}
	most := uint(l)
	for s, x := range lens {
		if x == 0 {
			continue
		}
		c := next[x]
		next[x]++
		if uint(x) <= tableBits || s < sym {
			continue
		}
		rev := int(bits.Reverse16(uint16(c)) >> (16 - uint(x))) // #nosec G115 -- c < 1<<15
		if rev&(1<<tableBits-1) == prefix {
			most = max(most, uint(x))
		}
	}
	return most - tableBits
}

func (d *inflater) buildLitlen(lens []uint8) bool {
	d.litlenNeed = shortest(lens)
	if len(lens) > 256 {
		d.litlenNeed = max(d.litlenNeed, int(lens[256]))
	}
	return build(d.litlen[:], lens, litlenBits, func(sym int, n uint32) uint32 {
		switch {
		case sym < 256:
			return hentry(kindLiteral, n, 0, uint32(sym)) // #nosec G115 -- small
		case sym == 256:
			return hentry(kindEnd, n, 0, 0)
		case sym < 286:
			return hentry(kindLength, n, lengthExtra[sym-257], lengthBase[sym-257])
		}
		return hentry(kindInvalid, n, 0, 0) // 286 and 287: corrupt when decoded
	})
}

func (d *inflater) buildDist(lens []uint8) bool {
	d.distNeed = shortest(lens)
	return build(d.dist[:], lens, distBits, func(sym int, n uint32) uint32 {
		if sym < 30 {
			return hentry(kindDist, n, distExtra[sym], distBase[sym])
		}
		return hentry(kindInvalid, n, 0, 0) // 30 and 31
	})
}

// bitReader is the input: a 64-bit buffer of nb bits, refilled eight
// bytes at a time from src[pos:]. Past the end it is fed zero bytes and
// pos keeps counting, so pos > len(src) means some of the buffer's bits
// are not in the stream; consumed says how many real ones were used.
type bitReader struct {
	src []byte
	pos int
	bb  uint64
	nb  uint
}

func (r *bitReader) refill() {
	if r.pos+8 <= len(r.src) {
		r.bb |= binary.LittleEndian.Uint64(r.src[r.pos:]) << r.nb
		r.pos += int(63-r.nb) >> 3
		r.nb |= 56
		return
	}
	for r.nb <= 56 {
		if r.pos < len(r.src) {
			r.bb |= uint64(r.src[r.pos]) << r.nb
		}
		r.pos++
		r.nb += 8
	}
}

// overrun reports whether the bits consumed so far go past the stream's
// end: then the stream was cut short.
func (r *bitReader) overrun() bool {
	return r.pos > len(r.src) && r.pos*8-int(r.nb) > len(r.src)*8
}

// avail is how many of the stream's real bits are left.
func (r *bitReader) avail() int {
	return len(r.src)*8 - (r.pos*8 - int(r.nb))
}

func (r *bitReader) take(n uint) uint32 {
	v := uint32(r.bb & (1<<n - 1)) // #nosec G115 -- n <= 13 bits
	r.bb >>= n
	r.nb -= n
	return v
}

// decode reads one symbol's entry from table, whose primary table has
// tableBits bits. The buffer must hold at least 15 bits.
func (r *bitReader) decode(table []uint32, tableBits uint) uint32 {
	e := table[r.bb&(1<<tableBits-1)]
	if e>>8&0xf == kindSub {
		r.bb >>= tableBits
		r.nb -= tableBits
		e = table[int(e>>16)+int(r.bb&(1<<(e>>12&0xf)-1))] // #nosec G115 -- at most 7 bits
	}
	n := uint(e & 31)
	r.bb >>= n
	r.nb -= n
	return e
}

func (d *inflater) inflate(dst, src []byte) ([]byte, error) {
	r := bitReader{src: src}
	out := dst[:cap(dst)]
	limit := len(dst)
	op := 0
	for {
		if op == limit {
			return out[:op], nil
		}
		r.refill()
		final := r.take(1)
		btype := r.take(2)
		if r.overrun() {
			return out[:op], nil
		}
		var t *inflater // the block's tables
		switch btype {
		case 0:
			var ok bool
			var err error
			if op, ok, err = d.stored(&r, out, op, limit); err != nil {
				return nil, err
			} else if !ok {
				return out[:op], nil // cut short
			}
			if final == 1 {
				return out[:op], nil
			}
			continue
		case 1:
			t = fixedTables()
		case 2:
			ok, err := d.header(&r)
			if err != nil {
				return nil, err
			}
			if !ok {
				return out[:op], nil // cut short
			}
			t = d
		default:
			return nil, errInflateCorrupt
		}
		var err error
		var end bool
		op, end, err = inflateBlock(&r, out, op, limit, t)
		if err != nil {
			return nil, err
		}
		if !end || final == 1 {
			return out[:op], nil // the limit, a stream cut short, or the last block
		}
	}
}

// stored copies a stored block. It reports false if the stream ends
// within it, having copied what there was, as compress/flate does.
func (d *inflater) stored(r *bitReader, out []byte, op, limit int) (int, bool, error) {
	r.take(r.nb & 7) // to a byte boundary
	// Hand the whole bytes still in the buffer back to the input.
	r.pos -= int(r.nb / 8)
	r.bb, r.nb = 0, 0
	if r.pos+4 > len(r.src) {
		return op, false, nil
	}
	n := int(binary.LittleEndian.Uint16(r.src[r.pos:]))
	if n != int(^binary.LittleEndian.Uint16(r.src[r.pos+2:])) {
		return op, false, errInflateCorrupt
	}
	r.pos += 4
	avail := min(n, len(r.src)-r.pos)
	c := copy(out[op:limit], r.src[r.pos:r.pos+avail])
	r.pos += avail
	op += c
	return op, avail == n || op == limit, nil
}

// header reads a dynamic block's code lengths and builds its tables. It
// reports false if the stream ends within the header.
func (d *inflater) header(r *bitReader) (bool, error) {
	r.refill()
	nlit := int(r.take(5)) + 257
	ndist := int(r.take(5)) + 1
	nclen := int(r.take(4)) + 4
	if r.overrun() {
		return false, nil
	}
	if nlit > 286 || ndist > 30 {
		return false, errInflateCorrupt
	}
	var clens [19]uint8
	for i := range nclen {
		if r.nb < 3 {
			r.refill()
		}
		clens[precodeOrder[i]] = uint8(r.take(3)) // #nosec G115 -- 3 bits
	}
	if r.overrun() {
		return false, nil
	}
	precodeNeed := shortest(clens[:])
	if !build(d.precode[:], clens[:], precodeBits, func(sym int, n uint32) uint32 {
		return hentry(kindLiteral, n, 0, uint32(sym)) // #nosec G115 -- small
	}) {
		return false, errInflateCorrupt
	}
	lens := d.lens[:nlit+ndist]
	for i := 0; i < len(lens); {
		if r.nb < 15+7 {
			r.refill()
		}
		if r.pos > len(r.src) && r.avail() < precodeNeed {
			return false, nil
		}
		e := r.decode(d.precode[:], precodeBits)
		if r.overrun() {
			return false, nil
		}
		if e>>8&0xf == kindInvalid {
			return false, errInflateCorrupt
		}
		sym := e >> 16
		if sym < 16 {
			lens[i] = uint8(sym) // #nosec G115 -- < 16
			i++
			continue
		}
		var rep int
		var val uint8
		switch sym {
		case 16:
			if i == 0 {
				return false, errInflateCorrupt
			}
			rep, val = 3+int(r.take(2)), lens[i-1]
		case 17:
			rep = 3 + int(r.take(3))
		default:
			rep = 11 + int(r.take(7))
		}
		if r.overrun() {
			return false, nil
		}
		if i+rep > len(lens) {
			return false, errInflateCorrupt
		}
		for range rep {
			lens[i] = val
			i++
		}
	}
	if !d.buildLitlen(lens[:nlit]) || !d.buildDist(lens[nlit:]) {
		return false, errInflateCorrupt
	}
	return true, nil
}

// block decodes one Huffman-coded block into out from op. end reports
// whether it reached the block's end; if not, it stopped at the limit or
// where the stream was cut short.
func inflateBlock(r *bitReader, out []byte, op, limit int, t *inflater) (int, bool, error) {
	litlen, dist := t.litlen[:], t.dist[:]
	op, end, err := fastBlock(r, out, op, limit, t)
	if end || err != nil {
		return op, end, err
	}
	// The fast path needs room for a whole match and its eight-byte
	// overshoot, and eight bytes of input for each refill.
	fastOut := limit - 258 - 8
	for {
		// Nothing past the limit is read, so nothing there can be an
		// error, as for a reader that stops at the limit.
		if op == limit {
			return op, false, nil
		}
		// One refill gives 56 bits: a length's code and extra bits
		// (20) and a distance's (28) fit.
		r.refill()
		if r.pos > len(r.src) && r.avail() < t.litlenNeed {
			return op, false, nil
		}
		e := r.decode(litlen, litlenBits)
		kind := e >> 8 & 0xf
		if kind == kindLiteral {
			if r.pos > len(r.src) && r.overrun() {
				return op, false, nil
			}
			out[op] = byte(e >> 16) // #nosec G115 -- a literal entry holds a byte
			op++
			continue
		}
		switch kind {
		case kindEnd:
			if r.pos > len(r.src) && r.overrun() {
				return op, false, nil
			}
			return op, true, nil
		case kindInvalid:
			if r.pos > len(r.src) && r.overrun() {
				return op, false, nil
			}
			return op, false, errInflateCorrupt
		}
		length := int(e>>16) + int(r.take(uint(e>>12&0xf)))
		if r.nb < 28 {
			r.refill()
		}
		if r.pos > len(r.src) && r.avail() < t.distNeed {
			return op, false, nil
		}
		e = r.decode(dist, distBits)
		if r.pos > len(r.src) && r.overrun() {
			return op, false, nil
		}
		if e>>8&0xf != kindDist {
			return op, false, errInflateCorrupt
		}
		distance := int(e>>16) + int(r.take(uint(e>>12&0xf)))
		if r.pos > len(r.src) && r.overrun() {
			return op, false, nil
		}
		if distance > op {
			return op, false, errInflateCorrupt
		}
		if op <= fastOut && distance >= 8 {
			// Eight bytes at a time; the source is always eight or
			// more bytes behind, so each word it reads is written.
			for i := 0; i < length; i += 8 {
				binary.LittleEndian.PutUint64(out[op+i:], binary.LittleEndian.Uint64(out[op-distance+i:]))
			}
			op += length
			continue
		}
		n := min(length, limit-op)
		from := op - distance
		for i := range n {
			out[op+i] = out[from+i]
		}
		op += n
		if op == limit {
			return op, false, nil
		}
	}
}

// fastBlock is inflateBlock's loop for while the input has eight bytes
// to refill from and the output room for a whole match and its
// overshoot, so that neither the stream's end nor the limit need
// checking. The bit reader lives in locals, which the compiler keeps in
// registers, rather than behind r. It returns with end false when it
// leaves that region, for inflateBlock to go on carefully.
func fastBlock(r *bitReader, out []byte, op, limit int, t *inflater) (int, bool, error) {
	const litlenMask, distMask = 1<<litlenBits - 1, 1<<distBits - 1
	src, pos, bb, nb := r.src, r.pos, r.bb, r.nb
	litlen, dist := &t.litlen, &t.dist
	fastOut := limit - 258 - 8
	le := binary.LittleEndian
	defer func() { r.pos, r.bb, r.nb = pos, bb, nb }()
	for pos+8 <= len(src) && op < fastOut {
		bb |= le.Uint64(src[pos:]) << nb
		pos += int(63-nb) >> 3
		nb |= 56
		e := litlen[bb&litlenMask]
		if e>>8&0xf == kindSub {
			bb >>= litlenBits
			nb -= litlenBits
			e = litlen[int(e>>16)+int(bb&(1<<(e>>12&0xf)-1))] // #nosec G115 -- at most 7 bits
		}
		bb >>= e & 31
		nb -= uint(e & 31)
		if e&0xf00 == kindLiteral<<8 {
			out[op] = byte(e >> 16) // #nosec G115 -- a literal entry holds a byte
			op++
			// At least 41 bits are left: a second literal fits.
			e = litlen[bb&litlenMask]
			if e&0xf00 == kindLiteral<<8 {
				bb >>= e & 31
				nb -= uint(e & 31)
				out[op] = byte(e >> 16) // #nosec G115 -- a literal entry holds a byte
				op++
			}
			continue
		}
		switch e >> 8 & 0xf {
		case kindEnd:
			return op, true, nil
		case kindInvalid:
			return op, false, errInflateCorrupt
		}
		n := uint(e >> 12 & 0xf)
		length := int(e>>16) + int(bb&(1<<n-1)) // #nosec G115 -- at most 5 bits
		bb >>= n
		nb -= n
		// 36 or more bits are left: a distance's code and extra fit.
		e = dist[bb&distMask]
		if e>>8&0xf == kindSub {
			bb >>= distBits
			nb -= distBits
			e = dist[int(e>>16)+int(bb&(1<<(e>>12&0xf)-1))] // #nosec G115 -- at most 7 bits
		}
		bb >>= e & 31
		nb -= uint(e & 31)
		if e>>8&0xf != kindDist {
			return op, false, errInflateCorrupt
		}
		n = uint(e >> 12 & 0xf)
		distance := int(e>>16) + int(bb&(1<<n-1)) // #nosec G115 -- at most 13 bits
		bb >>= n
		nb -= n
		if distance > op {
			return op, false, errInflateCorrupt
		}
		switch {
		case distance >= 8:
			// The source is at least eight bytes behind, so each word
			// it reads has been written.
			for i := 0; i < length; i += 8 {
				le.PutUint64(out[op+i:], le.Uint64(out[op-distance+i:]))
			}
		case distance == 1:
			v := uint64(out[op-1]) * 0x0101010101010101
			for i := 0; i < length; i += 8 {
				le.PutUint64(out[op+i:], v)
			}
		default:
			from := op - distance
			for i := range length {
				out[op+i] = out[from+i]
			}
		}
		op += length
	}
	return op, false, nil
}
