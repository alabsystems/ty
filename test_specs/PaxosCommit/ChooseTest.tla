---- MODULE ChooseTest ----
EXTENDS Integers

\* Test: CHOOSE n \in S : TRUE should return first element (in sorted order)
ChooseFirst(S) == CHOOSE n \in S : TRUE

VARIABLE x

Init ==
  x \in {ChooseFirst({0}), ChooseFirst({1}), ChooseFirst({0,1}), ChooseFirst({0,1,2})}

Next == UNCHANGED x

Spec == Init /\ [][Next]_x

\* All values should be the minimum of each set
AllValuesCorrect == x \in {0, 1}  \* {min({0}), min({1}), min({0,1}), min({0,1,2})} = {0, 1, 0, 0} = {0, 1}

====
