"""Pure readers for the two binary artifacts the runtime identity check inspects (A2.5, D99(7)).

- `elf_dynamic_symbols`: an ELF64 little-endian file's format facts and its dynamic symbol table (`.dynsym`), with each
  symbol's binding, type, visibility and whether it is defined. The table is found through the section headers and
  must sit where the dynamic segment's DT_SYMTAB and DT_STRTAB say, which is the table the loader uses.
- `usable_exports`: whether each required function is a defined, externally visible dynamic function export.
- `gguf_header`: a GGUF (version 2 or 3) file's metadata and tensor-type counts, read from the header only.

A structure these readers do not support raises `Unsupported` with the reason. That is never a pass: the caller
records the check as incomplete.
"""
from __future__ import annotations

import struct
from collections import Counter

EM_AARCH64, ET_DYN = 183, 3
SHT_STRTAB, SHT_DYNSYM, PT_DYNAMIC = 3, 11, 2
DT_NULL, DT_STRTAB, DT_SYMTAB = 0, 5, 6
BIND = {0: "LOCAL", 1: "GLOBAL", 2: "WEAK", 10: "GNU_UNIQUE"}
TYPE = {0: "NOTYPE", 1: "OBJECT", 2: "FUNC", 3: "SECTION", 4: "FILE", 5: "COMMON", 6: "TLS", 10: "GNU_IFUNC"}
VISIBILITY = {0: "DEFAULT", 1: "INTERNAL", 2: "HIDDEN", 3: "PROTECTED"}
GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
              11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 24: "I8", 25: "I16", 26: "I32", 27: "I64",
              28: "F64", 30: "BF16"}
FILE_TYPES = {0: "ALL_F32", 1: "MOSTLY_F16", 2: "MOSTLY_Q4_0", 3: "MOSTLY_Q4_1", 7: "MOSTLY_Q8_0", 8: "MOSTLY_Q5_0",
              9: "MOSTLY_Q5_1", 10: "MOSTLY_Q2_K", 15: "MOSTLY_Q4_K_M", 17: "MOSTLY_Q5_K_M", 18: "MOSTLY_Q6_K",
              32: "MOSTLY_BF16"}


class Unsupported(ValueError):
    """A structure the reader does not support or a file it cannot read: the check is incomplete, never a pass."""


def _unpack(fmt, data, offset, what):
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > len(data):
        raise Unsupported(f"truncated: {what} at byte {offset} lies outside the file")
    return struct.unpack_from(fmt, data, offset)


def elf_dynamic_symbols(data: bytes) -> dict:
    if data[:4] != b"\x7fELF":
        raise Unsupported("not an ELF file")
    if len(data) < 64:
        raise Unsupported("truncated ELF header")
    if data[4] != 2:
        raise Unsupported(f"ELF class {data[4]}: only ELF64 is supported")
    if data[5] != 1:
        raise Unsupported(f"ELF data encoding {data[5]}: only little-endian is supported")
    e_type, e_machine = _unpack("<HH", data, 16, "e_type")
    e_phoff, e_shoff = _unpack("<QQ", data, 32, "e_phoff")
    e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx = _unpack("<HHHHH", data, 54, "e_phentsize")
    facts = {"class": "ELF64", "endianness": "little", "machine": e_machine, "machine_is_aarch64": e_machine == EM_AARCH64,
             "type": e_type, "type_is_shared_object": e_type == ET_DYN}
    if e_shoff == 0 or e_shnum == 0:
        raise Unsupported("no section header table")
    if e_shentsize != 64 or e_phentsize not in (0, 56):
        raise Unsupported(f"unexpected header entry sizes (sections {e_shentsize}, segments {e_phentsize})")
    sections = []
    for k in range(e_shnum):
        sh = _unpack("<IIQQQQIIQQ", data, e_shoff + 64 * k, f"section header {k}")
        sections.append({"type": sh[1], "addr": sh[3], "offset": sh[4], "size": sh[5], "link": sh[6], "entsize": sh[9]})
    dynsyms = [s for s in sections if s["type"] == SHT_DYNSYM]
    if len(dynsyms) != 1:
        raise Unsupported(f"{len(dynsyms)} dynamic symbol tables (exactly one is supported)")
    dynsym = dynsyms[0]
    if dynsym["entsize"] != 24 or dynsym["size"] % 24:
        raise Unsupported("the dynamic symbol table's entry size is not 24 bytes")
    if not 0 <= dynsym["link"] < len(sections) or sections[dynsym["link"]]["type"] != SHT_STRTAB:
        raise Unsupported("the dynamic symbol table does not link a string table")
    strtab = sections[dynsym["link"]]
    dyn = {}
    for k in range(e_phnum):
        p_type, _, p_offset, p_vaddr, _, p_filesz = _unpack("<IIQQQQ", data, e_phoff + 56 * k, f"program header {k}")
        if p_type == PT_DYNAMIC:
            for j in range(p_filesz // 16):
                tag, val = _unpack("<qQ", data, p_offset + 16 * j, "dynamic entry")
                if tag == DT_NULL:
                    break
                dyn.setdefault(tag, val)
    if DT_SYMTAB not in dyn or DT_STRTAB not in dyn:
        raise Unsupported("no dynamic segment naming the symbol and string tables")
    if dyn[DT_SYMTAB] != dynsym["addr"] or dyn[DT_STRTAB] != strtab["addr"]:
        raise Unsupported("the section headers' symbol tables are not the ones the dynamic segment names")
    strings = data[strtab["offset"]:strtab["offset"] + strtab["size"]]
    if len(strings) != strtab["size"]:
        raise Unsupported("truncated dynamic string table")
    symbols = {}
    for k in range(1, dynsym["size"] // 24):
        st_name, st_info, st_other, st_shndx, st_value, st_size = _unpack("<IBBHQQ", data, dynsym["offset"] + 24 * k,
                                                                          f"dynamic symbol {k}")
        end = strings.find(b"\0", st_name)
        if st_name >= len(strings) or end < 0:
            raise Unsupported(f"dynamic symbol {k} has a name outside the string table")
        name = strings[st_name:end].decode("utf-8", "replace")
        if not name:   # unnamed (section) symbols cannot be looked up and are not listed
            continue
        symbols.setdefault(name, {"bind": BIND.get(st_info >> 4, str(st_info >> 4)),
                                  "type": TYPE.get(st_info & 15, str(st_info & 15)),
                                  "visibility": VISIBILITY[st_other & 3], "defined": st_shndx != 0,
                                  "value": st_value, "size": st_size})
    return {"format": facts, "dynamic_symbols": len(symbols), "symbols": symbols}


def usable_exports(symbols: dict, required) -> dict:
    """Each required name: usable when it is a defined dynamic FUNC with GLOBAL or WEAK binding and DEFAULT or
    PROTECTED visibility; otherwise the reasons. Other exports are counted, never a failure."""
    out = {}
    for name in required:
        s = symbols.get(name)
        if s is None:
            out[name] = {"usable": False, "reasons": ["not in the dynamic symbol table"]}
            continue
        reasons = [r for r, bad in (("undefined (imported, not exported)", not s["defined"]),
                                    (f"binding {s['bind']}", s["bind"] not in ("GLOBAL", "WEAK")),
                                    (f"visibility {s['visibility']}", s["visibility"] not in ("DEFAULT", "PROTECTED")),
                                    (f"type {s['type']}", s["type"] != "FUNC")) if bad]
        out[name] = {"usable": not reasons, "reasons": reasons, **{k: s[k] for k in ("bind", "type", "visibility")}}
    return out


class _Reader:
    def __init__(self, f, size):
        self.f, self.size = f, size

    def take(self, n, what):
        b = self.f.read(n)
        if len(b) != n:
            raise Unsupported(f"truncated GGUF: {what}")
        return b

    def u32(self, what):
        return struct.unpack("<I", self.take(4, what))[0]

    def u64(self, what):
        return struct.unpack("<Q", self.take(8, what))[0]

    def string(self, what):
        n = self.u64(what)
        if n > self.size:
            raise Unsupported(f"a string longer than the file: {what}")
        return self.take(n, what).decode("utf-8", "replace")


_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _value(r, vtype, what, depth=0):
    if vtype in _SCALARS:
        fmt = _SCALARS[vtype]
        return struct.unpack(fmt, r.take(struct.calcsize(fmt), what))[0]
    if vtype == 8:
        return r.string(what)
    if vtype == 9 and depth == 0:
        etype, count = r.u32(what), r.u64(what)
        if count > r.size:
            raise Unsupported(f"an array longer than the file: {what}")
        values = [_value(r, etype, what, 1) for _ in range(count)]
        return {"array_of": etype, "length": count, "first": values[:4]}
    raise Unsupported(f"unsupported GGUF value type {vtype}: {what}")


def gguf_header(path, details=False) -> dict:
    """The GGUF header of the file at path: version, metadata (arrays summarized) and tensor-type counts. With details,
    also every metadata value (arrays as summaries) and each tensor's name and shape. Only the header is read."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(0)
        r = _Reader(f, size)
        if r.take(4, "magic") != b"GGUF":
            raise Unsupported("not a GGUF file")
        version = r.u32("version")
        if version not in (2, 3):
            raise Unsupported(f"GGUF version {version}: versions 2 and 3 are supported")
        tensors, kvs = r.u64("tensor count"), r.u64("metadata count")
        if tensors > size or kvs > size:
            raise Unsupported("tensor or metadata counts larger than the file")
        meta = {}
        for k in range(kvs):
            key = r.string(f"metadata key {k}")
            meta[key] = _value(r, r.u32(f"type of {key}"), key)
        types, shapes = Counter(), {}
        for k in range(tensors):
            name = r.string(f"tensor {k} name")
            dims = r.u32(f"tensor {k} dimensions")
            if dims > 8:
                raise Unsupported(f"tensor {k} has {dims} dimensions")
            shapes[name] = [r.u64(f"tensor {k} shape") for _ in range(dims)]
            types[GGML_TYPES.get(r.u32(f"tensor {k} type"), "unknown")] += 1
            r.u64(f"tensor {k} offset")
    ft = meta.get("general.file_type")
    return {"version": version, "tensor_count": tensors, "metadata_count": kvs,
            "architecture": meta.get("general.architecture"), "name": meta.get("general.name"),
            "file_type": ft, "file_type_name": FILE_TYPES.get(ft) if isinstance(ft, int) else None,
            "quantization_version": meta.get("general.quantization_version"),
            "tensor_types": dict(sorted(types.items())),
            **({"metadata": meta, "tensor_shapes": shapes} if details else {})}
