import torch
import numpy as np
"""
t = np.array([[0, 1],
               [2, 3],
               [4,5]],
               )

t = np.array([[[[0, 1,2],
               [2, 3,4],
               [1,2,3]],
               [[0, 1,2],
               [2, 3,4],
               [1,2,3]]],
               [[[0, 1,2],
               [2, 3,4],
               [1,2,3]],
               [[0, 1,2],
               [2, 3,4],
               [1,2,3]]],
               [[[0, 1,2],
               [2, 3,4],
               [1,2,3]],
               [[0, 1,2],
               [2, 3,4],
               [1,2,3]]],
               [[[0, 1,2],
               [2, 3,4],
               [1,2,3]],
               [[0, 1,2],
               [2, 3,4],
               [1,2,3]]]]
               )
t1 = np.array([[[[0, 1,2],
               [2, 3,4],
               [1,2,3]],
               [[0, 1,2],
               [2, 3,4],
               [1,2,3]]],
               [[[0, 1,2],
               [2, 3,4],
               [1,2,3]],
               [[0, 1,2],
               [2, 3,4],
               [1,2,3]]]]
               )      
"""
"""               
a = torch.FloatTensor(t)
#b = torch.FloatTensor(t1)
print(t.shape)
print(t)
b=torch.eye(4)

print(b)
print(b.shape)

c = torch.stack([b,b,b],dim=2)
c = torch.stack([c,c,c],dim=3)
#c=c.numpy()
#c = b.repeat(4,2,1,1)
#c = torch.ones(4,2,3,3)
#a = torch.rand(4,2,1,1)
#a = torch.FloatTensor(t)
#b = torch.rand(4,2,1,1)
#w = torch.eye(3)
#c = b.matmul(a)
#d = np.dot(t,c)
#d = torch.dot(a,c)
#print(c)
#print(c.shape)
print(c)
print(c.shape)
for i in range(3):
    for j in range(3):
        tempw = c[:,:,i,j].squeeze(-1).squeeze(-1)
        print(tempw)
#print(d)
#print(d.shape)
#print(w)
#print(w.shape)
#print(a.shape)
#print(c.shape)

"""

def remove_p(path):
    testweight = torch.load(path)
    keys = list(testweight)
    for key in keys:
        print(key)
        if key.endswith('mw'):
            del(testweight[key])
        if key.endswith('2048'):
            del(testweight[key])
        if key.endswith('1024'):
            del(testweight[key])
        if key.endswith('512'):
            del(testweight[key])
        if key.endswith('W1'):
            del(testweight[key])
        if key.endswith('W2'):
            del(testweight[key])
        if key.endswith('W3'):
            del(testweight[key])

    remove_p = list(testweight)
    for rk in remove_p:
        print(rk)
    torch.save(testweight, 'remove_p.pth')

#lat_layers = (print("a") for x in range(10))
#print(lat_layers)
