from pyGraphics import *
import matplotlib.pyplot as plt
from copy import deepcopy

height = 10.05

#Scene init
#------------------------------------------
cam = Camera(5,1,5)
scene = Scene(1280,720, cam)

ground = Box(100,1,100)
ground.setColor((0,255,0))

egg = Sphere(0.25, Point3(0,height,0), 5)
egg.setColor((200,200,200))

scene.add(egg)
scene.add(ground)
cam.lookAt(egg.getPos())
#------------------------------------------

#Variables
y = height
v = 0
a = 0

#constants
m = 0.100 #kg
dragCoefficient = 0.145 #kg/m

fg = m * -9.81 #N
t = 0 #s


#Holders for graph info
a_s = []
v_s = []
y_s = []
t_s = []
f_s = []

while y > 0: 
    # Pipeline => F -> t -> a -> v -> y

    #Renderer handling
    dt = scene.getFrameTime()       # Get frame's delta t
    scene.frameStart()              # initiate frame

    #Physics
    fd = v * v * dragCoefficient    # get drag force
    sumF = fg + fd                  # get sum of forces
    t += dt                         # update time
    a = sumF / m                    # get acceleration
    v += a * dt                     # get new velocity
    y += v * dt                     # get new position

    #Add data to graph holders
    f_s.append(sumF)
    t_s.append(t) 
    y_s.append(y)
    v_s.append(v)

    #Scene management
    egg.moveToXYZ(0,y+0.25+0.5,0)
    cam.lookAt(egg.getPos())

    #end frame
    scene.frameEnd()

# vars for collision
j = 0
j_s = []
t2_s = []
a = v / 0.6 
dt = 1e-6
t2 = 0

# This section works off of the assumption that the egg
# experienced a uniform force over the entire impact.
# Realisticly, this is untrue, but we have no better methods
# or models.

while t2 < 0.06: #too quick for scene just done numerically 

    t2 += dt                # update new time
    f = m * a               # get force from known acceleration
    j += f * dt             # add force to numerical integral

    #data holders for graph
    j_s.append(j)
    t2_s.append(t2 + t)
    f_s.append(f)
    
#Data output
print(f"Final time: {t:.3f}s")
print(f"Final velocity: {v:.3f}m/s")
print(f"Max Energy: {0.5*v*v*m:.3f}J")
print(f"Max Force: {f:.3f}N")
print(f"Final height: {y:.3f}m")
print(f"Final Impulse: {j:.3f}Ns")


#combine t1 and t2 datasets for force
t_sTotal = deepcopy(t_s) #use deepcopy to avoid pointer issues
t_sTotal.extend(t2_s)

#Exit scene
sleep(1)
pygame.quit()

#Begin graphing
#--------------------------------------------

#Set to 1280x720 px
plt.figure(dpi=100, figsize=(12.80, 7.20))

#Position vs Time graph
plt.subplot(2,2,1)
plt.plot(t_s,y_s)
plt.title('Position vs Time')


#Velocity vs Time graph 
plt.subplot(2,2,2)
plt.plot(t_s,v_s)
plt.title('Velocity vs time')
plt.gca().invert_yaxis() #inverted to emphasize magnitude not direction


#Force vs Time
plt.subplot(2,2,3)
plt.plot(t_sTotal,f_s)
plt.title('Force vs time')
plt.gca().invert_yaxis() #inverted to emphasize magnitude not direction

#Impulse of collision vs time
plt.subplot(2,2,4)
plt.plot(t2_s,j_s)
plt.title('Impulse vs time (during impact)')
plt.gca().invert_yaxis() #inverted to emphasize magnitude not direction




plt.show()
#--------------------------------------------
