import random

t = int(input("t:"))
a, b = map(float, input().split())
c, d = map(float, input().split())
Wa,Wb,Tt = 0,0,0
for i in range(t):
    q = random.uniform(0, 0.9)
    Da, Db = 0, 0
    if q > a:
        Da = 1
    q = random.uniform(0, 0.9)
    if q > c:
        Db = 1
    if Da > Db:
        Wa +=1
    elif Da <Db:
        Wb +=1
    Tt += 1
print(Wa,Wb,Tt)
print(Wa/Tt, Wb/Tt)