#!/usr/bin/env bash
# What a full disk does to a 508 MB output, in the GDAL container
# (--privileged, for the loop mount): a 100 MB tmpfs written through the
# mapping (12 workers) and through WriteAt (1 worker), and a 100 MB ext4
# whose fallocate fails before any work. Each must end in an error, not a
# crash, and the ext4 file must be left empty.
#   docker run --rm --privileged --tmpfs /small:size=100m,exec \
#     -v <inputs>:/disk -v <dir with suite-new and this script>:/p <image> bash /p/diskfull.sh
A="-mode chunked -op slope -src raw -file /disk/prep/dem.raw -w 11264 -h 11264 -fill 65535 -cell 12.5 -tile 256"
echo "== tmpfs 100 MB, 12 workers (mapped)"; GOMAXPROCS=12 /p/suite-new $A -workers 12 -dir /small | grep -v '^op='; echo "exit=${PIPESTATUS[0]}"
echo "== tmpfs 100 MB, 1 worker (WriteAt)"; GOMAXPROCS=1 /p/suite-new $A -workers 1 -dir /small | grep -v '^op='; echo "exit=${PIPESTATUS[0]}"
command -v mkfs.ext4 >/dev/null || { apt-get -qq update >/dev/null && apt-get -qq install -y e2fsprogs >/dev/null; }
truncate -s 100M /tmp/img && mkfs.ext4 -q /tmp/img && mkdir -p /ext && mount -o loop /tmp/img /ext
echo "== ext4 100 MB, 12 workers"; GOMAXPROCS=12 /p/suite-new $A -workers 12 -dir /ext | grep -v '^op='; echo "exit=${PIPESTATUS[0]}"
echo "left behind: $(stat -c '%n %s bytes' /ext/strata-slope.raw)"
