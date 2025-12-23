
import math
z=3
q=4
P10=0.0008101511794681432
# after removing all deg10 nodes f=1
znew=(z-10*P10)/(1-P10)
kg2=12-90*P10 # numerator of factorial moment
fk=(kg2)/(1-P10)
qnew=fk/znew
qnew