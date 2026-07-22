--------------------------- MODULE AssumeOnlyCacheTest ---------------------------
(* Minimal spec to test ASSUME-only operator caching (#1031) *)
(* Author: Andrew Yates <andrewyates.name@gmail.com> *)

EXTENDS Naturals

(* Expensive operator that should be cached across multiple ASSUME calls *)
ExpensiveOp(n) == LET RECURSIVE Sum(_)
                      Sum(k) == IF k = 0 THEN 0 ELSE k + Sum(k-1)
                  IN Sum(n)

(* Multiple ASSUMEs calling the same operator with same arguments *)
ASSUME ExpensiveOp(10) = 55
ASSUME ExpensiveOp(10) = 55  \* Should hit cache
ASSUME ExpensiveOp(10) + ExpensiveOp(10) = 110  \* Both should hit cache

(* Different arguments - should compute and cache *)
ASSUME ExpensiveOp(5) = 15
ASSUME ExpensiveOp(5) = 15  \* Should hit cache

(* No state variables - ASSUME-only spec *)

=============================================================================
