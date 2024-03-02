bench on `Intel(R) Xeon(R) Gold 6146 CPU @ 3.20GHz`

```sh
❯ cargo bench
    Finished bench [optimized] target(s) in 0.01s
     Running unittests src/lib.rs (target/release/deps/bench1-127824086a45ebcb)

running 6 tests
test tests::test_dot ... ignored
test tests::test_mm512_2intersect_epi32 ... ignored
test tests::bench_dot                ... bench:         932 ns/iter (+/- 68)
test tests::bench_dot_avx            ... bench:         651 ns/iter (+/- 45)
test tests::bench_dot_avx_branchless ... bench:         689 ns/iter (+/- 29)
test tests::bench_dot_branchless     ... bench:       1,586 ns/iter (+/- 32)

test result: ok. 0 passed; 0 failed; 2 ignored; 4 measured; 0 filtered out; finished in 4.42s
```
