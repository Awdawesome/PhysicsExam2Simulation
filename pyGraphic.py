import math
import pygame
import moderngl
import numpy as np
from math import *
from copy import deepcopy
from time import time, sleep
#from Cs3D.gjk import check_intersection

#####################################################################################
#                                                                                   #
#  Program:    pyGraphics                                                           #
#                                                                                   #
#####################################################################################
#                                                                                   #
# Author: Andrew Doyle                                                              #
#                                                                                   #
#####################################################################################
#                                                                                   #
# Last revision: yes                                                                #
#                                                                                   #
#####################################################################################
#                                                                                   #
# Description:                                                                      #
#   Overview:                                                                       #
#            Runs and manages a 3D scene.                                           #
#                                                                                   #
#                                                                                   #
#   Objects and functions:                                                          #
#                                                                                   #
#       Classes:                                                                    #
#           ----------------------------------------------------------------------- #
#           | Vec3            | Implements a 3D vector                            | #   
#           |-----------------|---------------------------------------------------| #     
#           | Point3          | Implements a 3D point                             | #
#           |-----------------|---------------------------------------------------| #      
#           | Matrix          |Implements matrices                                | #
#           |-----------------|---------------------------------------------------| #                        
#           | Face            | Small face data holder                            | #
#           |-----------------|---------------------------------------------------| #
#           | Point2          | implements 2D points                              | #
#           |-----------------|---------------------------------------------------| #
#           | Poly            | Polygon data holder for OGL Renderer              | #
#           |-----------------|---------------------------------------------------| #
#           | KeyState        | Manages and maintains keystates                   | #                         | #
#           |-----------------|---------------------------------------------------| #
#           | Box             | A ThreeDimObject implementation that makes a      | #
#           |                 | rectangular prism of length x width x height      | #
#           |-----------------|---------------------------------------------------| #
#           | Mesh            | Creates a user-defined mesh of any shape from a   | #
#           |                 | list of Faces defined in world-space              | #
#           |-----------------|---------------------------------------------------| #
#           | RegularHedron   | Creates either a tetrahedron, an octohedron,      | #
#           |                 | a dodecahedron, or an icosohedron based on number | #
#           |                 | of sides specified                                | #
#           |-----------------|---------------------------------------------------| #
#           | NHedronPrism    | Creates a prism where the caps are regular n-gons | #
#           |-----------------|---------------------------------------------------| #
#           | Sphere          | Fairly expensive due to triangulation. Creates a  | #
#           |                 | sphere using triangular mesh                      | #
#           |-----------------|---------------------------------------------------| #
#           | Camera          | Virtual camera used to determine rendering        | #
#           |                 | properties, does not contain rendering logic      | #
#           |-----------------|---------------------------------------------------| #
#           | _GLRenderer     | Private: Object responsible for translating into  | #
#           |                 | OGL and shader handling; does the rendering       | #
#           |-----------------|---------------------------------------------------| #
#           | Scene           | Acts as a container for all rendered objects and  | #
#           |                 | Handles calling the _GLRenderer. Also handles     | #
#           |                 | user inputs and frame timing                      | #
#           |-----------------|---------------------------------------------------| #
#           | Collection      | Serves as a container for multiple objects.       | #
#           |                 | Allows multiple objects to be transformed at once | #
#           ----------------------------------------------------------------------- #
#                                                                                   #
#                                                                                   #
#                                                                                   #
#                                                                                   #
#                                                                                   #
#                                                                                   #
#                                                                                   #
#       Functions:                                                                  #
#               TODO: ADD FUNC DOCUMENTATION                                        #
#                                                                                   #
#####################################################################################

# ---------------- Helper Classes ----------------

# An implementation of 3D vectors
class Vec3:
    def __init__(self, x=0, y=0, z=0, otherOBJ=None):
        '''
        Creates a three dimensional vector of <x, y, z>.\n
        Option: otherOBJ: otherOBJ allows passing of object such as Point3 or list for Vec3 creation
        '''
        if(otherOBJ):
            if isinstance(otherOBJ, Point3):
                x,y,z = otherOBJ.x, otherOBJ.y, otherOBJ.z
            if isinstance(otherOBJ, list):
                x,y,z = otherOBJ[0], otherOBJ[1], otherOBJ[2]
            

        self._x = x
        self._y = y
        self._z = z
        self._magnitude = sqrt(x*x + y*y + z*z)
        
    def __iter__(self):
        yield self._x
        yield self._y
        yield self._z
    
    def normalize(self):
        '''Returns a new Vec3 in the same direction but with a magnitude of 1. If the original vector has a magnitude of 0, returns a zero vector.'''
        if(self._magnitude == 0):
            return Vec3(0,0,0)
        else:
            x = self._x / self._magnitude
            y = self._y / self._magnitude
            z = self._z / self._magnitude
            return Vec3(x, y, z)

    def magnitude(self):
        '''Returns the magnitude (length) of the vector'''
        return self._magnitude

    def __add__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self._x + other._x, self._y + other._y, self._z + other._z)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self._x - other._x, self._y - other._y, self._z - other._z)
        return NotImplemented
    
    def dot(self, other):
         '''
         Takes the dot product of the vector with another
         '''
         if isinstance(other, Vec3): 
            return self._x * other._x + self._y * other._y + self._z * other._z 
         else: return None 
    
    def cross(self, other): 
        '''
        Takes the cross product of the vector with another.\n
        The vector the method is called by is the left operand.\n
        I.e leftOperandVec.cross(rightOperandVec)
        '''
        if isinstance(other, Vec3): 
            x = self._y * other._z - self._z * other._y 
            y = -(self._x * other._z - self._z * other._x) 
            z = self._x * other._y - self._y * other._x 
            return Vec3(x,y,z) 
        return None

    def __mul__(self, other):
         '''
         Takes the dot product of the vector with another or mutliplies by the scalar
         '''
         if isinstance(other, Vec3): 
            return self._x * other._x + self._y * other._y + self._z * other._z 
         else:
             if (isinstance(other, int) or isinstance(other, float)):
                return Vec3(self._x * other, self._y * other, self._z * other)

    def __rmul__(self, other):
         '''
         Takes the dot product of the vector with another or mutliplies by the scalar
         '''
         if isinstance(other, Vec3): 
            return self._x * other._x + self._y * other._y + self._z * other._z 
         else:
             if (isinstance(other, int) or isinstance(other, float)):
                return Vec3(self._x * other, self._y * other, self._z * other)
        
    def __len__(self):
        '''Returns the number of components in the vector (always 3)'''
        return 3
    
    def getX(self):
        '''Returns the x component of the vector'''
        return self._x
    def getY(self):
        '''Returns the y component of the vector'''
        return self._y
    def getZ(self):
        '''Returns the z component of the vector'''
        return self._z
    

    def setX(self, x):
        '''Set the x component of the vector and update magnitude'''
        self._x = x
        self._magnitude = sqrt(self._x**2 + self._y**2 + self._z**2)
    def setY(self, y):
        '''Set the y component of the vector and update magnitude'''
        self._y = y
        self._magnitude = sqrt(self._x**2 + self._y**2 + self._z**2)
    def setZ(self, z):
        '''Set the z component of the vector and update magnitude'''
        self._z = z
        self._magnitude = sqrt(self._x**2 + self._y**2 + self._z**2)
   
    def __getitem__(self, key):
        if key == 'x' or key == 0:
            return self._x
        elif key == 'y' or key == 1:
            return self._y
        elif key == 'z' or key == 2:
            return self._z


    def bitWiseMult(self, other):
        '''Component-wise multiplication (Hadamard product)'''
        if isinstance(other, Vec3):
            return Vec3(self._x * other._x, self._y * other._y, self._z * other._z)
        return NotImplemented

    def __repr__(self):
        return(f"<{self._x}, {self._y}, {self._z}>")

# A 3D point class
class Point3:
    def __init__(self, x=0, y=0, z=0, otherOBJ = None):
        '''Creates a three dimensional point of (x, y, z).\n
        Option: otherOBJ: otherOBJ allows passing of object such as Vec3 or list for Point3 creation
        '''

        if(otherOBJ):
            if isinstance(otherOBJ, Vec3):
                x,y,z = otherOBJ._x, otherOBJ._y, otherOBJ._z
            if isinstance(otherOBJ, list):
                x,y,z = otherOBJ[0], otherOBJ[1], otherOBJ[2]

        self.x = x
        self.y = y
        self.z = z
    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return(Point3(x,y,z))
    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z
        return(Point3(x,y,z))
    
    def __truediv__(self, other):
        x = self.x / other
        y = self.y / other
        z = self.z / other
        return(Point3(x,y,z))
    
    def __mul__(self, other):
        x = self.x * other
        y = self.y * other
        z = self.z * other
        return(Point3(x,y,z))
    
    def __getitem__(self, key):
        if key == 'x' or key == 0:
            return self.x
        elif key == 'y' or key == 1:
            return self.y
        elif key == 'z' or key == 2:
            return self.z
    
    def __repr__(self):
        return f'({self.x}, {self.y}, {self.z})'

# A matrix class  
class Matrix:
    def __init__(self, rows):
        '''
        Create a matrix with the rows specified
        Rows should be arrays of the same length containing either padding data or the true data
        '''
        if not rows or not all(len(r) == len(rows[0]) for r in rows):
            raise ValueError("All rows must have the same number of columns.")
        
        self.rows = [list(r) for r in rows]
        self.numberOfRows = len(rows)
        self.numberOfColumns = len(rows[0])

    def __getitem__(self, idx):
        return self.rows[idx]

    def multiply_matrix(self, other):
        """Multiply this matrix (self) by another matrix (other)"""
        if self.numberOfColumns != other.numberOfRows:
            raise ValueError("Incompatible dimensions for matrix multiplication")
        result = [[0]*other.numberOfColumns for _ in range(self.numberOfRows)]
        for i in range(self.numberOfRows):
            for j in range(other.numberOfColumns):
                for k in range(self.numberOfColumns):
                    result[i][j] += self.rows[i][k] * other.rows[k][j]
        return Matrix(result)
    
    def multiply_vector(self, vec):
        """Multiply this matrix (self) by a vector"""
        if self.numberOfColumns != len(vec):
            raise ValueError("Incompatible dimensions for vector multiplication")
        return Vec3(otherOBJ=[sum(self.rows[i][j] * vec[j] for j in range(self.numberOfColumns)) for i in range(self.numberOfRows)])
    
    def skew(v):
        """Return a skew-symmetric matrix from a 3-element vector"""
        x, y, z = v
        return Matrix([[0, -z, y],
                       [z, 0, -x],
                       [-y, x, 0]])

# A small Face interface
class Face:
    def __init__(self, point1 : Point3, point2 : Point3, point3 : Point3):
        self._points = (point1, point2, point3)
    def getPoints(self):
        return self._points

# A 2D point class
class Point2:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def getX(self):
        return self._x

    def getY(self):
        return self._y

# an abstract for OGL renderer
class Poly:
    def __init__(self):
        self.pts = []
        self.color = (255, 255, 255)
        self.width = 0

    def clearPoints(self):
        self.pts = []

    def addPoint(self, p):
        self.pts.append((p.getX(), p.getY()))

    def setFillColor(self, c):
        self.color = c

    def setBorderWidth(self, w):
        self.width = w

# Holds info on keypresses
class KeyState:
    def __init__(self):
        self._keysPressed = set()
        self._edges = {}

    def pressKey(self, k): 
        self._keysPressed.add(k)
    
    def releaseKey(self, k): 
        self._keysPressed.discard(k)

    def keyEdgeDown(self, k): 
        self._edges[k] = "down"

    def keyEdgeUp(self, k): 
        self._edges[k] = "up"

    def getKeysPressed(self):
        return self._keysPressed
    
    def getKeyEdges(self):
        t = dict(self._edges)
        self._edges.clear()
        return t

#--------------------------------------------
#3D Objects
#--------------------------------------------

# -------------- Framework -------------
# A 3D object base class that implements overall internals
class Mesh:
    def __init__(self, faces=None, color=(100,100,100)):
        '''
        base mesh class
        '''

        # --- Transform / state (from ThreeDimObject) ---
        self._color = color
        self._center = Point3()
        self._rotMatrix = Matrix([[1,0,0],[0,1,0],[0,0,1]])
        self._scale = 1
        self._hasMoved = True
        self._tags = {}
        self._centerAdjustment = Point3()

        # --- Geometry ---
        self._localPoints = []
        self._points = []
        self._faces = []
        self._poly = []

        if faces:
            self._buildFromFaces(faces)

    # =========================
    # GEOMETRY BUILDING
    # =========================

    def _buildFromFaces(self, faces):
        keyMap = {}

        for face in faces:
            newFace = []

            for point in face.getPoints():
                if point not in keyMap:
                    keyMap[point] = len(self._localPoints)
                    self._localPoints.append(point)

                newFace.append(keyMap[point])

            self._faces.append(tuple(newFace))

        self._computePoints()
        self._buildPolyList()
    def _buildPolyList(self):
        self._poly = [Poly() for _ in self._faces]

    def _computePoints(self):
        points = []

        for point in self._localPoints:
            scaled = point * self._scale + self._centerAdjustment
            rotated = self._rotMatrix.multiply_vector(Vec3(otherOBJ=scaled))
            translated = Point3(otherOBJ=[
                rotated[i] + self._center[i] for i in range(3)
            ])
            points.append(translated)

        self._points = points

    # =========================
    # Transforms
    # =========================

    def moveXYZ(self, x=0, y=0, z=0):
        self._center += Point3(x,y,z)
        self._computePoints()
        self._hasMoved = True

    def moveToXYZ(self, x=0, y=0, z=0):
        self._center = Point3(x,y,z)
        self._computePoints()
        self._hasMoved = True

    def moveVec(self, deltaVec=Vec3(0,0,0)):
        self._center += Point3(otherOBJ=deltaVec)
        self._computePoints()
        self._hasMoved = True

    def moveToVec(self, deltaVec=Vec3(0,0,0)):
        self._center = Point3(otherOBJ=deltaVec)
        self._computePoints()
        self._hasMoved = True

    def scale(self, scale):
        self._scale *= scale
        self._computePoints()

    def rotate(self, x=0, y=0, z=0):
        self._hasMoved = True

        rx, ry, rz = radians(x), radians(y), radians(z)

        cosx, sinx = cos(rx), sin(rx)
        cosy, siny = cos(ry), sin(ry)
        cosz, sinz = cos(rz), sin(rz)

        Rx = Matrix([
            [1,0,0],
            [0,cosx,-sinx],
            [0,sinx,cosx]
        ])

        Ry = Matrix([
            [cosy,0,siny],
            [0,1,0],
            [-siny,0,cosy]
        ])

        Rz = Matrix([
            [cosz,-sinz,0],
            [sinz,cosz,0],
            [0,0,1]
        ])

        self._rotMatrix = Rz.multiply_matrix(Ry).multiply_matrix(Rx).multiply_matrix(self._rotMatrix)

        self._computePoints()

    def adjustCenter(self, p : Point3):

        if not isinstance(p, Point3):
            try:
                p = Point3(otherOBJ=p)
            except:
                try:
                    p = Point3(p[0],p[1],p[2])
                except:
                    raise(TypeError, "Invalid Type for adjust center")
            self._centerAdjustment = p
            self._computePoints()

    # =========================
    # GETTERS
    # =========================

    def getCenter(self):
        return self._center
    
    def _getPoints(self):
        return self._points
    
    def _getFaces(self):
        return self._faces
    
    
    # =========================
    # UTIL
    # =========================

    def ptsToNumpy(self):
        return np.array([[p.x,p.y,p.z] for p in self._points], dtype=np.float64)

    #def collidesWith(self, otherObj):
    #    return check_intersection(self.ptsToNumpy(), otherObj.ptsToNumpy())

    def setColor(self, color):
        self._color = color

    def getPos(self):
        return self._center

    def getRotMat(self):
        return self._rotMatrix

    def getRotDegrees(self):
        m = self._rotMatrix

        m00,m01,m02 = m[0]
        m10,m11,m12 = m[1]
        m20,m21,m22 = m[2]

        pitch = math.asin(-m20)

        if abs(m20) < 0.9999:
            yaw = math.atan2(m10, m00)
            roll = math.atan2(m21, m22)
        else:
            yaw = math.atan2(-m01, m11)
            roll = 0

        return (degrees(yaw), degrees(pitch), degrees(roll))

    # =========================
    # TAG SYSTEM
    # =========================

    def addTag(self, key, val):
        self._tags[key] = val

    def removeTag(self, key):
        self._tags.pop(key, None)

    def getTagVal(self, key):
        return self._tags.get(key)   
# ----------------- Box ----------------
# A 3D object implementation that forms a rectangular prism of lxwxh
class Box(Mesh):
    def __init__(self, length=1, width=1, height=1, center=Point3(0,0,0)):
        '''
        Creates a box of length by width by height at the 3D point \n
        (Measurements are from farthest points on the same edge, length would not be center to edge, but edge to edge) \n
        Center is the center of mass of the box and thus inside the box
        '''
        super().__init__()
        if not isinstance(center, Point3):
            raise TypeError("center must be a Point3 object")
        
        self._length = length
        self._width = width
        self._height = height

        self._center = center

        self._color = (0,0,0)


        self._rotMatrix = Matrix([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ])

        self._points = []
        self._computePoints()

        self._faces = [
            [0,2,3,1],  # bottom
            [4,5,7,6],  # top
            [0,1,5,4],  # front
            [2,6,7,3],  # back
            [0,4,6,2],  # left
            [1,3,7,5]   # right
        ]
    
    def getDimensions(self):
        '''Returns the dimensions of the box as a tuple of (length, width, height)'''
        return (self._length, self._width, self._height)

    def rotate(self, x=0, y=0, z=0):
        '''Rotate using Euler input but store as matrix'''
        self._hasMoved = True

        rx = radians(x)
        ry = radians(y)
        rz = radians(z)

        cosx, sinx = cos(rx), sin(rx)
        cosy, siny = cos(ry), sin(ry)
        cosz, sinz = cos(rz), sin(rz)

        Rx = [
            [1,    0,     0],
            [0, cosx, -sinx],
            [0, sinx,  cosx]
        ]

        Ry = [
            [ cosy, 0, siny],
            [    0, 1,    0],
            [-siny, 0, cosy]
        ]

        Rz = [
            [cosz, -sinz, 0],
            [sinz,  cosz, 0],
            [   0,     0, 1]
        ]

        def matmul(a, b):
            return [
                [
                    a[i][0]*b[0][j] +
                    a[i][1]*b[1][j] +
                    a[i][2]*b[2][j]
                    for j in range(3)
                ]
                for i in range(3)
            ]

        # Same order as before: X → Y → Z
        newR = matmul(Rz, matmul(Ry, Rx))

        # Compose with existing rotation
        self._rotMatrix = Matrix(matmul(newR, self._rotMatrix))

        self._computePoints()
        self._hasMoved = True
        



    def _computePoints(self):
        '''PRIVATE: Recompute all raw points using stored rotation matrix'''
        dx, dy, dz = self._length/2, self._width/2, self._height/2
        cx, cy, cz = self._center.x, self._center.y, self._center.z

        points = [
            Point3(cx - dx, cy - dy, cz - dz),
            Point3(cx + dx, cy - dy, cz - dz),
            Point3(cx - dx, cy + dy, cz - dz),
            Point3(cx + dx, cy + dy, cz - dz),
            Point3(cx - dx, cy - dy, cz + dz),
            Point3(cx + dx, cy - dy, cz + dz),
            Point3(cx - dx, cy + dy, cz + dz),
            Point3(cx + dx, cy + dy, cz + dz)
        ]

        R = self._rotMatrix

        def apply_rotation(p):
            x, y, z = p.x - cx, p.y - cy, p.z - cz

            rx = R[0][0]*x + R[0][1]*y + R[0][2]*z
            ry = R[1][0]*x + R[1][1]*y + R[1][2]*z
            rz = R[2][0]*x + R[2][1]*y + R[2][2]*z

            return Point3(rx + cx, ry + cy, rz + cz)

        self._points = [apply_rotation(p) for p in points]

#Creates either a tetrahedron, an octohedron, a dodecahedron, or an icosohedron 
class RegularHedron(Mesh):
    """A regular polyhedron (Platonic solid) with n-sided faces.
    
    Supports 4/5 regular polyhedra (Cube is Box class):
    - Tetrahedron (4 faces, 3 sides each)
    - Octahedron (8 faces, 3 sides each)
    - Dodecahedron (12 faces, 5 sides each)
    - Icosahedron (20 faces, 3 sides each)
    """
    
    def __init__(self, sides, scale=1.0):
        """Create a regular polyhedron.
        
        Args:
            sides (int): Number of sides per face
                - 3: Icosahedron (20 triangular faces)
                - 4: Cube (6 square faces)
                - 5: Dodecahedron (12 pentagonal faces)
                Note: Tetrahedron and Octahedron also have 3 sides but are 
                distinguished by their structure
            scale (float): Scale factor for the polyhedron
        """
        self.sides = sides
        self.scale = scale
        
        if sides == 4:
            faces = self._createTetrahedron()
        elif sides == 8:
            faces = self._createOctahedron()
        elif sides == 12:
            faces = self._createDodecahedron()
        elif sides == 20:
            faces = self._createIcosahedron()
        else:
            raise ValueError("Only 3, 4, or 5 sided regular polyhedra exist")
        
        super().__init__(faces)
        
    
    def _createTetrahedron(self):
        """Create a regular tetrahedron (4 triangular faces)"""
        a = 1.0 / sqrt(3)
        vertices = [
            Point3(a, a, a) * self.scale,
            Point3(a, -a, -a) * self.scale,
            Point3(-a, a, -a) * self.scale,
            Point3(-a, -a, a) * self.scale,
        ]
        
        faces = [
            Face(vertices[0], vertices[1], vertices[2]),
            Face(vertices[0], vertices[2], vertices[3]),
            Face(vertices[0], vertices[3], vertices[1]),
            Face(vertices[1], vertices[3], vertices[2]),
        ]
        return faces
    
    def _createOctahedron(self):
        """Create a regular octahedron (8 triangular faces)"""
        s = self.scale
        vertices = [
            Point3(s, 0, 0),
            Point3(-s, 0, 0),
            Point3(0, s, 0),
            Point3(0, -s, 0),
            Point3(0, 0, s),
            Point3(0, 0, -s),
        ]
        
        faces = [
            Face(vertices[0], vertices[2], vertices[4]),
            Face(vertices[0], vertices[4], vertices[3]),
            Face(vertices[0], vertices[3], vertices[5]),
            Face(vertices[0], vertices[5], vertices[2]),
            Face(vertices[1], vertices[4], vertices[2]),
            Face(vertices[1], vertices[3], vertices[4]),
            Face(vertices[1], vertices[5], vertices[3]),
            Face(vertices[1], vertices[2], vertices[5]),
        ]
        return faces
    
    def _createDodecahedron(self):
        """Create a regular dodecahedron (12 pentagonal faces)"""
        from math import sqrt
        
        phi = (1 + sqrt(5)) / 2  # Golden ratio
        s = self.scale
        
        vertices = [
            # (±1, ±1, ±1)
            Point3(s, s, s),           # 0
            Point3(s, s, -s),          # 1
            Point3(s, -s, s),          # 2
            Point3(s, -s, -s),         # 3
            Point3(-s, s, s),          # 4
            Point3(-s, s, -s),         # 5
            Point3(-s, -s, s),         # 6
            Point3(-s, -s, -s),        # 7
            # (0, ±1/φ, ±φ)
            Point3(0, s/phi, s*phi),   # 8
            Point3(0, s/phi, -s*phi),  # 9
            Point3(0, -s/phi, s*phi),  # 10
            Point3(0, -s/phi, -s*phi), # 11
            # (±1/φ, ±φ, 0)
            Point3(s/phi, s*phi, 0),   # 12
            Point3(s/phi, -s*phi, 0),  # 13
            Point3(-s/phi, s*phi, 0),  # 14
            Point3(-s/phi, -s*phi, 0), # 15
            # (±φ, 0, ±1/φ)
            Point3(s*phi, 0, s/phi),   # 16
            Point3(s*phi, 0, -s/phi),  # 17
            Point3(-s*phi, 0, s/phi),  # 18
            Point3(-s*phi, 0, -s/phi), # 19
        ]
        
        # Correct 12 pentagonal faces
        pentagons = [
            [0, 8, 4, 14, 12],
            [0, 16, 2, 10, 8],
            [0, 12, 1, 17, 16],
            [1, 12, 14, 5, 9],
            [1, 9, 11, 3, 17],
            [2, 16, 17, 3, 13],
            [2, 13, 15, 6, 10],
            [4, 8, 10, 6, 18],
            [4, 18, 19, 5, 14],
            [3, 11, 7, 15, 13],
            [5, 19, 7, 11, 9],
            [6, 15, 7, 19, 18],
        ]
        
        faces = []
        for pentagon in pentagons:
            # Triangulate each pentagon from first vertex
            for i in range(1, len(pentagon) - 1):
                faces.append(Face(
                    vertices[pentagon[0]],
                    vertices[pentagon[i]],
                    vertices[pentagon[i + 1]]
                ))
        
        return faces
    
    def _createIcosahedron(self):
        """Create a regular icosahedron (20 triangular faces)"""
        phi = (1 + sqrt(5)) / 2  # Golden ratio
        s = self.scale
        
        vertices = [
            Point3(-s, s*phi, 0),
            Point3(s, s*phi, 0),
            Point3(-s, -s*phi, 0),
            Point3(s, -s*phi, 0),
            Point3(0, -s, s*phi),
            Point3(0, s, s*phi),
            Point3(0, -s, -s*phi),
            Point3(0, s, -s*phi),
            Point3(s*phi, 0, -s),
            Point3(s*phi, 0, s),
            Point3(-s*phi, 0, -s),
            Point3(-s*phi, 0, s),
        ]
        
        faces = [
            # 5 faces around point 0
            Face(vertices[0], vertices[11], vertices[5]),
            Face(vertices[0], vertices[5], vertices[1]),
            Face(vertices[0], vertices[1], vertices[7]),
            Face(vertices[0], vertices[7], vertices[10]),
            Face(vertices[0], vertices[10], vertices[11]),
            # 5 adjacent faces
            Face(vertices[1], vertices[5], vertices[9]),
            Face(vertices[5], vertices[11], vertices[4]),
            Face(vertices[11], vertices[10], vertices[2]),
            Face(vertices[10], vertices[7], vertices[6]),
            Face(vertices[7], vertices[1], vertices[8]),
            # 5 faces around point 3
            Face(vertices[3], vertices[9], vertices[4]),
            Face(vertices[3], vertices[4], vertices[2]),
            Face(vertices[3], vertices[2], vertices[6]),
            Face(vertices[3], vertices[6], vertices[8]),
            Face(vertices[3], vertices[8], vertices[9]),
            # 5 adjacent faces
            Face(vertices[4], vertices[9], vertices[5]),
            Face(vertices[2], vertices[4], vertices[11]),
            Face(vertices[6], vertices[2], vertices[10]),
            Face(vertices[8], vertices[6], vertices[7]),
            Face(vertices[9], vertices[8], vertices[1]),
        ]
        return faces

#creates a prism where the caps are regular n-gons 
class NHedronPrism(Mesh):
    """A prism where the top and bottom are regular n-gons, 
    connected by rectangular side faces at a given height."""
    
    def __init__(self, sides, radius=1.0, height=2.0):
        """
        Create a new nHedronPrism.
        
        Args:
            sides (int): Number of sides for the regular n-gon base/top (n >= 3)
            radius (float): Circumradius of the n-gon
            height (float): Height of the prism (distance between top and bottom)
        """
        if sides < 3:
            raise ValueError("sides must be >= 3")
        
        self.sides = sides
        self.radius = radius
        self.height = height
        
        # Generate faces for the prism
        faces = self._generateFaces()
        
        # Call parent constructor
        super().__init__(faces)
        self.rotate(0,0,180/sides-90)
    
    def _generateFaces(self):
        """Generate the faces of the prism (all triangulated)"""
        faces = []
        
        # Create top and bottom n-gons
        top_vertices = self._createNGon(self.radius, self.height / 2)
        bottom_vertices = self._createNGon(self.radius, -self.height / 2)
        
        # Triangulate top face (n-gon)
        for i in range(1, len(top_vertices) - 1):
            faces.append(Face(top_vertices[0], top_vertices[i], top_vertices[i + 1]))
        
        # Triangulate bottom face (n-gon, reversed for correct normal)
        bottom_rev = list(reversed(bottom_vertices))
        for i in range(1, len(bottom_rev) - 1):
            faces.append(Face(bottom_rev[0], bottom_rev[i], bottom_rev[i + 1]))
        
        # Side faces (rectangles split into 2 triangles each)
        for i in range(self.sides):
            next_i = (i + 1) % self.sides
            
            # Rectangle vertices
            p1 = bottom_vertices[i]
            p2 = bottom_vertices[next_i]
            p3 = top_vertices[next_i]
            p4 = top_vertices[i]
            
            # Split into 2 triangles
            faces.append(Face(p1, p2, p3))
            faces.append(Face(p1, p3, p4))
        
        return faces
    
    def _createNGon(self, radius, z_height):
        """Create vertices for a regular n-gon at given height"""
        vertices = []
        for i in range(self.sides):
            angle = 2 * pi * i / self.sides
            x = radius * cos(angle)
            y = radius * sin(angle)
            vertices.append(Point3(x, y, z_height))
        return vertices
#----------------- Sphere ---------------

#creates an extremely laggy sphere
class Sphere(Mesh):
    def __init__(self, r, center=Point3(0,0,0), subdiv=4):
        """
        WARNING EXTREMELY LAGGY!!! 4 subdivisions = 512 faces, 8 subdivisions = 2048 faces, 16 subdivisions = 8192 faces, etc.\n
        1 sphere reduces fps by ~50%\n
        Triangular mesh sphere\n
        r: radius\n
        subdiv: resolution (higher = smoother)
        """
        super().__init__([Face(Point3(0,0,0), Point3(0,0,0), Point3(0,0,0),)])  # dummy face to satisfy constructor

        stacks = max(3, subdiv)
        slices = max(6, subdiv * 2)

        points = [[None for _ in range(slices + 1)] for _ in range(stacks + 1)]
        faces = []

        # Generate sphere vertices
        for i in range(stacks + 1):
            theta = pi * i / stacks  # 0 -> pi
            sin_t = sin(theta)
            cos_t = cos(theta)

            for j in range(slices + 1):
                phi = 2 * pi * j / slices  # 0 -> 2pi
                sin_p = sin(phi)
                cos_p = cos(phi)

                x = r * sin_t * cos_p
                y = r * cos_t
                z = r * sin_t * sin_p

                points[i][j] = Point3(x, y, z)

        # Build triangular faces
        for i in range(stacks):
            for j in range(slices):
                p1 = points[i][j]
                p2 = points[i][j + 1]
                p3 = points[i + 1][j]
                p4 = points[i + 1][j + 1]

                # two triangles per grid cell
                faces.append(Face(p1, p2, p3))
                faces.append(Face(p2, p4, p3))

        super().__init__(faces)
        self._buildPolyList()
        

        self.moveToXYZ(center.x, center.y, center.z)

# ---------------- Camera ----------------
# Camera for scene, doesnt do rendering itself but has useful properties for rendering
class Camera:
    def __init__(self, x=0, y=0, z=0, rotationMatrix : Matrix = None, fov=90, nearClip=0.1, farClip=200):
        '''Create a camera object at (z,y,z) looking in the direction of the rotationMatrix (default looking at +z)'''
        self._x = x
        self._y = y
        self._z = z
        self._yaw = 0
        self._pitch = 0
        self._roll = 0
        self._hasMoved = True
        self._near = nearClip
        self._far = farClip

        self.fov = fov


        if rotationMatrix is None:
            self.rot = Matrix([[1,0,0],[0,1,0],[0,0,1]])
        else:
            self.rot = rotationMatrix

    def setYawPitch(self, yaw, pitch):
        '''Set the camera rotation using yaw and pitch angles (in degrees). If None is passed for either, that angle will not be changed.'''
        if(yaw == None):
            yaw = self._yaw
        if(pitch == None):
            pitch = self._pitch
        self._yaw = yaw
        self._pitch = pitch
        self._buildCameraRotation()

    def getYawPitch(self):
        '''Returns the camera rotation as a tuple of (yaw, pitch) angles in degrees'''
        return (self._yaw, self._pitch)

    def deltaYawPitch(self, deltaYaw, deltaPitch):
        '''Change the camera rotation by the given delta yaw and pitch angles (in degrees).'''
        self._yaw += deltaYaw
        self._pitch += deltaPitch
        self._buildCameraRotation()

    def moveXYZ(self, x=0, y=0, z=0):
        '''Move the camera relative to its current pos'''
        self._x += x
        self._y += y
        self._z += z
        self._hasMoved = True

    def moveToXYZ(self, x=0, y=0, z=0):
        '''Move the camera to the new pos'''
        self._x = x
        self._y = y
        self._z = z
        self._hasMoved = True
    
    def moveVec(self, deltaVec : Vec3 = Vec3(0,0,0)):
        '''Move the camera relative to its current pos'''

        if isinstance(deltaVec, Point3):
            deltaVec = Vec3(otherOBJ=deltaVec)
        if isinstance(deltaVec, list):
            deltaVec = Vec3(otherOBJ=deltaVec)

        self._x += deltaVec._x
        self._y += deltaVec._y
        self._z += deltaVec._z
        self._hasMoved = True

    def moveToVec(self, vec : Vec3 = Vec3(0,0,0)):
        '''Move the camera to the new pos'''

        if isinstance(vec, Point3):
            vec = Vec3(otherOBJ=vec)
        if isinstance(vec, list):
            vec = Vec3(otherOBJ=vec)

        self._x = vec._x
        self._y = vec._y
        self._z = vec._z
        self._hasMoved = True

    def lookAt(self, target=Point3(0,0,0)):
        '''Set the camera rotation to look at a target point in world space'''


        forward = (Vec3(target.x, target.y, target.z) -
                Vec3(self._x, self._y, self._z)).normalize()

        worldUp = Vec3(0, 1, 0)

        right = worldUp.cross(forward)
        rMag = sqrt(right._x**2 + right._y**2 + right._z**2)

        if rMag < 1e-6:
            right = Vec3(1, 0, 0)
            rMag = 1

        right = Vec3(right._x / rMag, right._y / rMag, right._z / rMag)

        trueUp = forward.cross(right)

        self.rot = Matrix([
            [right._x,  right._y,  right._z],
            [trueUp._x, trueUp._y, trueUp._z],
            [forward._x, forward._y, forward._z]
        ])
  
    def getRightVector(self):
        '''Returns the right vector of the camera as a Vec3'''
        return Vec3(self.rot[0][0], self.rot[0][1], self.rot[0][2])

    def getUpVector(self):
        '''Returns the up vector of the camera as a Vec3'''
        return Vec3(self.rot[1][0], self.rot[1][1], self.rot[1][2])
    
    def getForwardVector(self):
        '''Returns the forward vector of the camera as a Vec3'''
        return Vec3(self.rot[2][0], self.rot[2][1], self.rot[2][2])

    def getPosition(self):
        '''Returns the position of the camera as a Vec3'''
        return Vec3(self._x, self._y, self._z)
    
    def isInView(self, point : Point3):
        '''Returns true if the point is in the camera's field of view'''
        camPos = Vec3(self._x, self._y, self._z)
        toPoint = Vec3(point.x, point.y, point.z) - camPos
        forward = self.getForwardVector()
        angle = acos(forward.dot(toPoint.normalize()))
        return angle < radians(self.fov / 2)

    def getFOV(self):
        '''Returns the field of view of the camera in degrees'''
        return self.fov
    
    def setFOV(self, fov):
        '''Sets the field of view of the camera in degrees'''
        self.fov = fov

    def setRotationMatrix(self, rotMatrix):
        '''Sets the rotation matrix of the camera'''
        self.rot = rotMatrix




    def _buildCameraRotation(self):
        '''PRIVATE: Build the camera rotation matrix from the current yaw and pitch angles'''
        yaw = radians(self._yaw)
        pitch = radians(self._pitch)

        cy, sy = cos(yaw), sin(yaw)
        cp, sp = cos(pitch), sin(pitch)

        # yaw (Y axis - turn left/right)
        Ry = [
            [ cy, 0, sy],
            [  0, 1,  0],
            [-sy, 0, cy]
        ]

        # pitch (X axis - look up/down)
        Rx = [
            [1,  0,   0],
            [0, cp, -sp],
            [0, sp,  cp]
        ]

        def matmul(a, b):
            return [
                [
                    a[i][0]*b[0][j] +
                    a[i][1]*b[1][j] +
                    a[i][2]*b[2][j]
                    for j in range(3)
                ]
                for i in range(3)
            ]

        # IMPORTANT: yaw first, then pitch
        self.rot = matmul(Rx, Ry)

# GL renderer. I love GL so much. *I lied*
class _GLRenderer:
    def __init__(self, pygame_surface, scene, near=0.1, far=200):

        self.surface = pygame_surface
        self.scene = scene
        self.ctx = scene.ctx
        self.nearClip = near
        self.farClip = far

        w, h = self.surface.get_size()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        self.subdivision_level = 2  # Increase for more detail (higher performance cost)
        self.light_vector = np.array([1.0, 100.0, 0], dtype='f4')
        self.light_vector = self.light_vector / np.linalg.norm(self.light_vector)  # Normalize

        self.vertex_shader_source = """
                #version 330
                
                in vec3 in_pos;
                in vec3 in_color;
                in vec3 in_normal;
                
                uniform mat4 projection;
                uniform mat4 view;
                uniform mat4 model;
                uniform vec3 light_dir;
                
                out vec3 v_color;
                out float v_brightness;
                
                void main() {
                    gl_Position = projection * view * model * vec4(in_pos, 1.0);
                    
                    // Transform normal to world space
                    vec3 world_normal = normalize(mat3(model) * in_normal);
                    
                    // Calculate Lambert shading (diffuse lighting)
                    float brightness = max(dot(world_normal, light_dir), 0.0);

                    brightness = pow(brightness, 0.3);
                    
                    // Add ambient light to prevent complete darkness
                    brightness = brightness * 0.8 + 0.2;
                    
                    v_color = in_color;
                    v_brightness = brightness;
                }
            """

        self.fragment_shader_source = """
                #version 330
                
                in vec3 v_color;
                in float v_brightness;
                out vec4 out_color;
                
                void main() {
                    out_color = vec4(v_color * v_brightness, 1.0);
                }
            """

        self.prog = self.ctx.program(
                vertex_shader=self.vertex_shader_source,
                fragment_shader=self.fragment_shader_source,
            )
        self.vbo = None
        self.vao = None
        self.frame_count = 0
        self.vertex_normals = None

    def _calculateVertexNormals(self, vertices, triangles):
        """Calculate per-vertex normals by averaging triangle normals"""
        # Initialize vertex normals
        vertex_normals = np.zeros_like(vertices)
        
        # Calculate triangle normals
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        triangle_normals = np.cross(edge1, edge2)
        norms = np.linalg.norm(triangle_normals, axis=1, keepdims=True)
        triangle_normals = triangle_normals / (norms + 1e-8)  # Normalize
        
        # Accumulate triangle normals to each vertex
        for i, triangle in enumerate(triangles):
            for vertex_idx in triangle:
                vertex_normals[vertex_idx] += triangle_normals[i]
        
        # Normalize all vertex normals
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_normals = vertex_normals / (norms + 1e-8)
        
        return vertex_normals

    def _buildBuffers(self):
        if len(self.scene.vertices) == 0:
            return

        vertices = np.array(self.scene.vertices, dtype="f4")
        triangles = np.array(self.scene.triangles, dtype="i4")
        colors = np.array(self.scene.colors, dtype="f4")  # Shape: (612, 3)

        if len(triangles) == 0:
            return

        vertex_normals = self._calculateVertexNormals(vertices, triangles)

        # Expand vertices by triangle indices
        flat_vertices = vertices[triangles].reshape(-1, 3)
        
        # CORRECT: Repeat each color 3 times (once per vertex in the triangle)
        flat_colors = np.repeat(colors, 3, axis=0) / 255.0  # Shape: (1836, 3)
        
        flat_normals = vertex_normals[triangles].reshape(-1, 3)

        # Interleave vertex, color, and normal data
        data = np.zeros(len(flat_vertices), dtype=[('position', 'f4', 3), ('color', 'f4', 3), ('normal', 'f4', 3)])
        data['position'] = flat_vertices
        data['color'] = flat_colors
        data['normal'] = flat_normals

        if self.vbo is None:
            self.vbo = self.ctx.buffer(data.tobytes())
            self.vao = self.ctx.vertex_array(
                self.prog,
                [(self.vbo, '3f 3f 3f', 'in_pos', 'in_color', 'in_normal')]
            )
        else:
            self.vbo.write(data.tobytes())

    def _render(self):
        self.frame_count += 1
        
        self._buildBuffers()

        self.ctx.viewport = (0, 0, *self.surface.get_size())
        self.ctx.clear(0.15, 0.15, 0.15, depth=1.0)

        if self.vao is None:
            return

        cam = self.scene.getCamera()
        w, h = self.surface.get_size()
        aspect = w / h if h > 0 else 1.0

        # Simple perspective projection
        fov_rad = np.radians(cam.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        near, far = self.nearClip, self.farClip

        proj = np.array([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, -(far+near)/(far-near), -1],
            [0, 0, -2*far*near/(far-near), 0],
        ], dtype='f4')

        # Simple view matrix from camera position and direction
        forward = cam.getForwardVector()
        right = cam.getRightVector()
        up = cam.getUpVector()
        pos = np.array([cam._x, cam._y, cam._z], dtype='f4')

        view = np.array([
            [right._x, up._x, -forward._x, 0],
            [right._y, up._y, -forward._y, 0],
            [right._z, up._z, -forward._z, 0],
            [-np.dot([right._x, right._y, right._z], pos),
             -np.dot([up._x, up._y, up._z], pos),
             np.dot([forward._x, forward._y, forward._z], pos),
             1],
        ], dtype='f4')

        # Identity model matrix (geometry already in world space)
        model = np.identity(4, dtype='f4')

        self.prog['projection'].write(proj.tobytes())
        self.prog['view'].write(view.tobytes())
        self.prog['model'].write(model.tobytes())
        self.prog['light_dir'].write(self.light_vector.tobytes())

        self.vao.render()

# ---------------- Scene ----------------
# The main object, contains all 3D objects, manages OGL renderer, manages input, and timings
class Scene:
    def __init__(self, w, h, camera):
        """
        camera: Camera object
        """

        pygame.init()

        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                        pygame.GL_CONTEXT_PROFILE_CORE)
        
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        self._canvas = pygame.display.set_mode((w,h), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        self.ctx = moderngl.create_context(require=330)
        self._camera = camera

        self._objects = []
        self._keyState = KeyState()

        self._renderer = _GLRenderer(self._canvas, self, near=self._camera._near, far=self._camera._far)

        # mouse
        self._mouseDown = False
        self._mouseX = 0
        self._mouseY = 0
        self._lastMouseX = 0
        self._lastMouseY = 0
        self._mouseDX = 0
        self._mouseDY = 0
        self._frameRate = 240


        self._frameStart = 0
        self._dtFrame = 0
        self._gameTime = 0

        self.vertices = []   # list or np array (N,3)
        self.triangles = []  # list or np array (M,3)
        self.colors = []     # list or np array (M,3) in 0–255 or 0–1

        # control flags
        self._mouseLock = False
        self._hasEscaped = False

    def setMouseLock(self, lock):
        pygame.event.set_grab(lock)
        pygame.mouse.set_visible(not lock)
        self._mouseLock = lock

    def isMouseDown(self):
        return self._mouseDown

    def getMousePosition(self):
        return (self._mouseX, self._mouseY)

    def getMouseDelta(self):
        dx, dy = self._mouseDX, self._mouseDY
        self._mouseDX = 0
        self._mouseDY = 0
        return (dx, dy)

    def getKeysPressed(self):
        return self._keyState.getKeysPressed()

    def getKeyEdges(self):
        return self._keyState.getKeyEdges()

    def isEscaped(self):
        return self._hasEscaped

    def add(self, obj):

        if isinstance(obj, Collection):
            for obj in obj.getObjects():
                self.add(obj)
            return
        

        self._objects.append(obj)

        pts = obj._getPoints()
        faces = obj._getFaces()

        if len(pts) == 0 or len(faces) == 0:
            return

        base_index = len(self.vertices)

        # -------------------------------------------------
        # append vertices
        # -------------------------------------------------
        for p in pts:
            self.vertices.append([p.x, p.y, p.z])

        # -------------------------------------------------
        # append triangles (with index offset)
        # -------------------------------------------------
        for f in faces:

            # support quad or triangle faces safely
            if len(f) < 3:
                continue

            # triangulate quads if needed
            for i in range(1, len(f) - 1):
                self.triangles.append([
                    base_index + f[0],
                    base_index + f[i],
                    base_index + f[i + 1]
                ])

                # assign object color per triangle
                self.colors.append(obj._color)
        
    def remove(self, obj):
        self._objects.remove(obj)

    def clear(self):
        self._objects = []

    def getObjects(self):
        return self._objects

    def getCamera(self):
        return self._camera

    def getCanvas(self):
        return self._canvas

    def objCount(self):
        return len(self._objects)

    def frameStart(self):
        self._processEvents()
        self._frameStart = time()

    def getFrameTime(self):    
        return self._dtFrame
    
    def getGameTime(self):
        return self._gameTime

    def frameEnd(self):
        self.render()
        self._dtFrame = time() - self._frameStart
        self._gameTime += self._dtFrame

    def render(self):
        self._rebuildGeometry()
        
        # Ensure viewport matches current window size
        w, h = self._canvas.get_size()
        self._renderer.ctx.viewport = (0, 0, w, h)
        
        self._renderer._render()
        pygame.display.flip()

    def delayTillEndOfFrame(self):
        t = time()
        t0 = self._frameStart
        target = 1/self._frameRate

        dt = t - t0
        tr = target - dt
        sleep(tr if tr > 0 else 0)

    def setFrameRate(self, fps):
        self._frameRate = fps

    def _processEvents(self):
        self._mouseDX = 0
        self._mouseDY = 0
        
        # Reset mouse to center ONCE per frame (before reading new events)
        if self._hasEscaped:
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)
            pygame.mouse.get_rel()  # flush

        for event in pygame.event.get():

            # KEY DOWN
            if event.type == pygame.KEYDOWN:
                key = pygame.key.name(event.key).lower()

                if key == "escape":
                    self._hasEscaped = not self._hasEscaped
                    pygame.mouse.set_visible(self._hasEscaped)
                    pygame.event.set_grab(not self._hasEscaped)
                    continue

                self._keyState.pressKey(key)
                self._keyState.keyEdgeDown(key)

            # KEY UP
            elif event.type == pygame.KEYUP:
                key = pygame.key.name(event.key).lower()
                self._keyState.releaseKey(key)
                self._keyState.keyEdgeUp(key)

            # MOUSE MOTION
            elif event.type == pygame.MOUSEMOTION:
                self._mouseDX, self._mouseDY = event.rel
                self._mouseDX, self._mouseDY = -self._mouseDX, -self._mouseDY 
            # MOUSE DOWN
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self._mouseDown = True

            # MOUSE UP
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self._mouseDown = False
            elif event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.VIDEORESIZE:
                w, h = event.size
                self._canvas = pygame.transform.scale(self._canvas, (w,h))
                #self._recreateContext(w, h)
                self._renderer.surface = self._canvas

    def _rebuildGeometry(self):
        self.vertices = []
        self.triangles = []
        self.colors = []

        for obj in self._objects:

            pts = obj._getPoints()
            faces = obj._getFaces()

            if not pts or not faces:
                continue

            base_index = len(self.vertices)

            for p in pts:
                self.vertices.append([p.x, p.y, p.z])

            for f in faces:
                if len(f) < 3:
                    continue

                for i in range(1, len(f) - 1):
                    self.triangles.append([
                        base_index + f[0],
                        base_index + f[i],
                        base_index + f[i + 1]
                    ])
                    self.colors.append(obj._color)


# -------------- Collection -------------
# Groups objects together for easier rotation and movement
class Collection:
    def __init__(self):
        '''Create an empty collection of 3D objects'''
        self._objects = []
        self._originals = []
        self._center = Point3(0,0,0)
        self._calculateCOM()
        self._rotation = Matrix([[1,0,0],[0,1,0],[0,0,1]])

    def add(self, object : Mesh):
        '''Add a 3D object to the collection'''
        if not isinstance(object, Mesh):
            raise TypeError('Parameter "object" must be a 3D object')
        self._objects.append(object)
        
        self._originals.append(deepcopy(object))  # snapshot original state
        self._calculateCOM()

    def scale(self, scale : int):

        #get local collection coords
        for obj in self._objects:
            localCoord = obj.getPos() - self._center
            localCoord *= scale
            obj.scale(scale)
            obj.moveToVec(Vec3(otherOBJ=localCoord))

    def moveXYZ(self, x=0, y=0, z=0):
        '''Move the entire collection by (x,y,z) relative to its current position'''
        for obj in self._objects:
            obj.moveXYZ(x,y,z)
      
    def rotate(self, x=0, y=0, z=0):
        '''Rotate the entire collection by (x,y,z) degrees around its center point'''
        rx, ry, rz = radians(x), radians(y), radians(z)
        R = _rotationMatrixXYZ(rx, ry, rz)

        self._rotation = self._rotation.multiply_matrix(R)

        cx, cy, cz = self._center.x, self._center.y, self._center.z

        for i, obj in enumerate(self._objects):
            original = self._originals[i]

            # Rotate position
            rel = Vec3(otherOBJ=original._getCenter()) - Vec3(cx, cy, cz)
            rotated = self._rotation.multiply_vector(rel)
            obj._rotMatrix = self._rotation.multiply_matrix(original._rotMatrix)
            obj.moveToXYZ(rotated[0] + cx, rotated[1] + cy, rotated[2] + cz)

    def remove(self, object : Mesh):
        '''Remove a 3D object from the collection'''
        if not isinstance(object, Mesh):
            raise TypeError('Parameter "object" must be a 3D object')
        self._objects.remove(object)
        index = self._originals.index(object)
        self._originals.pop(index)
        self._calculateCOM()
    
    def getObjects(self):
        '''Returns a list of all objects in the collection'''
        return self._objects
    
    def getCenter(self):
        '''Returns the center point of the collection'''
        return self._center
    
    def getRotation(self):
        '''Returns the current rotation matrix of the collection'''
        return self._rotation
    
    def getOriginals(self):
        '''Returns a list of the original states of all objects in the collection'''
        return self._originals
    
    def getObjectCount(self):
        '''Returns the number of objects in the collection'''
        return len(self._objects)
    
    def clear(self):
        '''Clear all objects from the collection'''
        self._objects = []
        self._originals = []
        self._center = Point3(0,0,0)
        self._rotation = Matrix([[1,0,0],[0,1,0],[0,0,1]])




    def _calculateCOM(self):
        '''PRIVATE: Calculate the center of mass (average position) of all objects in the collection'''
        avgPoint = Point3(0,0,0)
        count = 0
        for object in self._originals:
            avgPoint += object.getCenter()
            count += 1
        if(count > 0):
            self._center = avgPoint / count
        else:
            self._center = Point3(0,0,0)
          

#----------------- Utility Functions ----------------

# Technically redundant, but looks nicer in code
def clone3D(obj):
    '''Returns a deep copy of the object. Useful for instancing multiple copies of an object with different transformations.'''
    return deepcopy(obj)

# converts (r, theta) -> (x,y)
def polarToCartesian(r, theta):
    '''Convert 2D polar (r,theta) into 2D Cartesian (x,y) points
    returns a tuple of (x,y)'''
    return (r*cos(theta), r*sin(theta))

# converts (x, y) -> (r,theta)
def cartesianToPolar(x, y):
    '''Convert 2D cartesian (x,y) into 2D polar (r,theta) points
    returns a tuple of (r,theta)'''
    return ((x*x + y*y)**0.5, atan(y/x))

#small clamp implementation
def clamp(value, minVal, maxVal):
    '''Clamp a value between a minimum and maximum'''
    return max(minVal, min(maxVal, value))


#Read and parse an obj into a polyhedron
def meshFromOBJ(file):
    try:
        with open(file) as obj:

            lines = obj.readlines()
            verts, faces = [], []
            
            for line in lines:
                
                if(line.startswith('f ')):
                    indecesDirty = line.split()
                    indecesDirty.pop(0) #discard designator

                    #get only the vertex indice
                    indeces = [int(pointDirty.split('/')[0]) - 1 for pointDirty in indecesDirty]
                    

                    if(len(indeces) > 3):
                        raise SyntaxError(f"INVALID FILE!\n\n\
                                File: {file}, is not triangulated!\n \
                                Please triangulate obj to continue.")
                    faces.append(indeces) #Only need vertex indices

                elif(line.startswith('v ')):
                    points = line.split()
                    x,y,z = points[1:4] #ignore designator and optional w arg
                    x,y,z = float(x), float(y), float(z)
                    verts.append(Point3(x,y,z))

            meshFaces = [Face(verts[i], verts[i1], verts[i2]) for i, i1, i2 in faces]
            return Mesh(meshFaces)

                

    except(FileNotFoundError):
        raise ValueError(f"INVALID FILE!\n\n\
            File: {file}, does not exist within context.\n \
            Try using absolute paths or a path relative to the lib")
    except(TypeError):
        raise SyntaxError(f"INVALID FILE!\n\n\
            File: {file}, is not formatted as expected!\n\
            Please ensure it is an obj and that it is formatted correctly")
    except:
        raise SyntaxError(f"Unkown Error!\n\n\
            File: {file}, experienced an unkown error\n\
            Please ensure it is an obj and that it is formatted correctly")


# Make a rot matrix from xyz rotation
def _rotationMatrixXYZ(phi, theta, psi):
    """Return 3x3 rotation matrix for XYZ Euler angles"""
    cx, sx = cos(phi), sin(phi)
    cy, sy = cos(theta), sin(theta)
    cz, sz = cos(psi), sin(psi)

    Rx = Matrix([[1, 0, 0],
                 [0, cx, -sx],
                 [0, sx, cx]])
    Ry = Matrix([[cy, 0, sy],
                 [0, 1, 0],
                 [-sy, 0, cy]])
    Rz = Matrix([[cz, -sz, 0],
                 [sz, cz, 0],
                 [0, 0, 1]])

    return Rz.multiply_matrix(Ry).multiply_matrix(Rx)

