import numpy as np
import matplotlib.pyplot as plt

# %%

r = 1.0  # circle radius

p = np.array([0.5, 0.7])  # initial position (x, y)
v = np.array([1.0, -0.4])  # initial velocity (vx, vy)

T = 21.  # total simulation time


# %%

# Your code here

def reflection_vector(velocity, normal):
    return velocity - 2 * np.dot(velocity, normal) * normal


# %%
def tick_rate_method(p, v, T):
    timeStamp = 0.00001
    steps = int(T / timeStamp)

    path = np.zeros((steps, 2))
    path[0] = p

    for step in range(1, steps):
        p += v * timeStamp

        distance_to_origin = np.linalg.norm(p)
        if distance_to_origin >= 1.0:
            normal = p / distance_to_origin
            v = reflection_vector(v, normal)  # Vector v after collision

        path[step] = p

    return path


def mathematic(p, v, T):
    positions = np.zeros((int(T + 1), 2))
    positions[0] = p
    for i in range(1, int(T + 1)):
        a = v[1] / v[0]
        b = p[1] - a * p[0]

        A = 1 + np.power(a, 2)
        B = 2 * a * b
        C = np.power(b, 2) - np.power(r, 2)

        x1 = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
        x2 = (-B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
        y1 = a * x1 + b
        y2 = a * x2 + b
        print(f'Point 1 : ({x1},{y1})')
        print(f'Point 2 : ({x2},{y2})')
        # Let's not talk about it :)
        if T == 20:
            if v[0] > 0 and v[1] > 0:
                if x1 < p[0]:
                    p = [x1, y1]
                else:
                    p = [x2, y2]
            elif v[0] > 0 and v[1] < 0:
                if x1 > p[0]:
                    p = [x1, y1]
                else:
                    p = [x2, y2]
            elif v[0] < 0 and v[1] > 0:
                if x1 < p[0]:
                    p = [x1, y1]
                else:
                    p = [x2, y2]
            elif v[0] < 0 and v[1] < 0:
                if x1 > p[0]:
                    p = [x1, y1]
                else:
                    p = [x2, y2]
        else:
            if np.isclose(p[0], x1, rtol=1e-05, atol=1e-08, equal_nan=False) and np.isclose(p[1], y1, rtol=1e-05,atol=1e-08,equal_nan=False):
                p = [x2, y2]
            else:
                p = [x1, y1]
        distance_to_origin = np.linalg.norm(p)
        positions[i] = p
        normal = p / distance_to_origin
        v = reflection_vector(v, normal)  # Vector v after collision
        print(f'End Velocity {v}')
        print(f'End Position {p}')
    print("----------------------------------")
    print(positions)
    return positions


print("Chose method:\n1.Tick rate (Slower) \n2.Mathematic (Doesn't work for T = 20)")
choice = int(input("Your choice: "))
if choice == 1:
    path = tick_rate_method(p, v, T)
    plt.title(f"Tick Rate method T = {T}")
else:
    path = mathematic(p, v, T)
    plt.title(f"Mathematic method T = {T}\n Doesn't work only for T = 20")

# Plot the unit circle
theta = np.linspace(0, 2 * np.pi, 1000)
circle_x = np.cos(theta)
circle_y = np.sin(theta)

plt.plot(circle_x, circle_y, label="Unit Circle")

plt.plot(path[:, 0], path[:, 1], label="Particle Path", color='red')

plt.scatter(path[0, 0], path[0, 1], color='black', marker='o')
plt.scatter(path[-1, 0], path[-1, 1], color='green', marker='o')


plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
plt.axis('equal')
plt.show()
