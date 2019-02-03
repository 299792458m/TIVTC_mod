# TIVTC_mod
a clone of pinterf's TIVTC

and added something
-TFMのslow=2の時の挙動を1.05互換に修正(多分1.0.11のバグ 条件が一部違うのと、prvnnf,nxtnnfのアドレス演算忘れ)
-OpenMPで無理やりforを回して速度改善(本質的にはメモリアクセスが非効率なためだが・・・)
