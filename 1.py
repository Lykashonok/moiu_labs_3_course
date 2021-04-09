import numpy as np
c = np.array([-2, -7, 1, 0])
A = np.array([[1, -6, 1, 0], [0, -5, 1, 1]])
b = np.array([-6., -10.])
Jb = np.array([3, 4])
iteration = 0

def generate_bazis(A, Jb):
  z = np.eye(len(Jb))
  j = 0
  for i in Jb:
    z[:, j] = A[:, i - 1]
    j += 1
  return z


def solve(c, A, b, Jb):
  global iteration
  while True:
    iteration += 1
    Ab = generate_bazis(A, Jb)
    Ab_inv = np.linalg.inv(Ab)
    cb = np.array([c[i - 1] for i in Jb])
    if iteration == 1:
      y = np.dot(cb, Ab_inv)
    xb = np.dot(Ab_inv, b)
    print("Ab")
    print(Ab)
    print("baslineKappa")
    print(xb)
    x = np.array([0 for i in range(len(A[0]))])
    for i in range(len(Jb)):
      x[Jb[i] - 1] = xb[i]
    x = x.astype(float)
    print("Kappa")
    print(x)
    if np.min(x) >= 0:
      return x
    for i in range(len(x)):
      if x[i] < 0:
        k = i + 1
    for i in range(len(Jb)):
      if Jb[i] == k:
        jk = i
    print("k")
    print(k)
    print("jk")
    print(jk)
    deltay = Ab_inv[jk]
    print("\n\n\nDeltay\n")
    print(Ab_inv)
    print(deltay)
    print("\n\n\n")
    Jn = []
    for i in range(1, len(x) + 1):
      if i not in Jb:
        Jn += [i]
    Jn = np.array(Jn)
    u = np.copy(Jn)
    u = u.astype(float)
    for i in range(len(u)):
      u[i] = np.dot(deltay, A[:, Jn[i] - 1])
    print("mu")
    print(u)
    print("Jn")
    print(Jn)
    if np.min(u) >= 0:
      return 'Задача несовместна'
    sigma = np.copy(u)
    sigma = sigma.astype(float)
    for i in range(len(sigma)):
      if u[i] >= 0:
        sigma[i] = 999999
      else:
        print("\n\n\n")
        sigma[i] = (c[Jn[i] - 1] - np.dot(A[:, Jn[i] - 1], y)) / u[i]
        print(A)
        print(Jn, Jn[i] - 1)
        print(A[:, Jn[i] - 1])
        print(y)
        print(np.dot(A[:, Jn[i] - 1], y))
        print("\n\n\n")
    sigma0 = np.min(sigma)

    j0 = Jn[np.argmin(sigma)]
    Jb[jk] = j0
    y += np.dot(sigma0, deltay)
    print("min sigma0")
    print(sigma0)
    print("Jb")
    print(Jb)
    print("y")
    print(y)

print(solve(c, A, b, Jb))
print("-----------")
