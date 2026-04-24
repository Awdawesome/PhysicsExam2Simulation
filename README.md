# PhysicsExam2Simulation
Simulation for the egg drop for Exam 2 in Phys 1610


Mathematic Derivation:
$v_t = \sqrt{\frac{mg}{k}}$

$v_t^2 = \frac{mg}{k}$

$kv_t^2 = mg$

$k = \frac{mg}{v_t^2}$


Since our item hit terminal velocity at sim time 0.503s, we can assume
$v_{avg} \approx v_t$


$V_{avg} = \frac{\Delta v}{\Delta t} = 2.597m/s$

$k = \frac{mg}{(2.597m/s)^2} = 0.145kg/m$




$\sum F = ma$

$\sum F = F_d - F_g$

$F_d - F_g = ma$

$F_g = mg$

$F_d = kv^2$

$kv^2 - mg = ma$

$\frac{kv^2}{m} - g = a$

$a(v) = \frac{0.145kg/m * v^2}{m} - g$


