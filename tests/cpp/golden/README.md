# Byte-parity fixtures

Each directory is one case: the inputs as raw little-endian arrays
(`col.<name>.f32`, `icol.<name>.i32`, `prop.<name>.f64`, `iprop.<name>.i64`,
`box.f64`, `origin.f64`, `atom_*`), the compression options in `meta.txt`, the
`.mdz` file the reference writer produced for them (`out.mdz`) and an FNV-1a
hash of the positions it decoded back (`decoded_fnv1a` in `meta.txt`).

`test_golden.cpp` compresses the same inputs with the same options and checks
the bytes match, and decodes each `out.mdz` and checks the hash. Changing an
encoder changes these bytes; for a deliberate change, regenerate them in the
same commit:

    MDZ_REGEN_GOLDEN=1 build/tests/cpp/mdz_tests golden_compress_is_byte_identical
