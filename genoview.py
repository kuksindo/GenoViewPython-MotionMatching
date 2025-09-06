from pyray import (
    Vector2, Vector3, Vector4, Transform, Matrix, Camera3D, 
    Color, Rectangle, Model, ModelAnimation, Mesh, BoneInfo, 
    Texture, RenderTexture)
from raylib import *

import os
import time
import bvh
import quat
import numpy as np
import struct
import cffi
ffi = cffi.FFI()

#----------------------------------------------------------------------------------
# Profile
#----------------------------------------------------------------------------------

class timewith:
    def __init__(self, name): self.name = name
    def __enter__(self): self.start = time.time()
    def __exit__(self, type, value, traceback): print('%s: %6.4fms' % (self.name, 1000 * (time.time() - self.start)))

def timefunc(f):
    def timedf(*args, **kwargs):
        start = time.time()
        out = f(*args, **kwargs)
        print('%s: %6.4fms' % (f.__name__, 1000 * (time.time() - start)))
        return out
    return timedf

#----------------------------------------------------------------------------------
# Camera
#----------------------------------------------------------------------------------

class Camera:

    def __init__(self):
        self.cam3d = Camera3D()
        self.cam3d.position = Vector3(2.0, 3.0, 5.0)
        self.cam3d.target = Vector3(-0.5, 1.0, 0.0)
        self.cam3d.up = Vector3(0.0, 1.0, 0.0)
        self.cam3d.fovy = 45.0
        self.cam3d.projection = CAMERA_PERSPECTIVE
        self.azimuth = 0.0
        self.altitude = 0.4
        self.distance = 4.0
        self.offset = Vector3Zero()
    
    def update(
        self,
        target,
        azimuthDelta,
        altitudeDelta,
        offsetDeltaX,
        offsetDeltaY,
        mouseWheel,
        dt):

        self.azimuth = self.azimuth + 1.0 * dt * -azimuthDelta
        self.altitude = Clamp(self.altitude + 1.0 * dt * altitudeDelta, 0.0, 0.4 * PI)
        self.distance = Clamp(self.distance +  20.0 * dt * -mouseWheel, 0.1, 100.0)
        
        rotationAzimuth = QuaternionFromAxisAngle(Vector3(0, 1, 0), self.azimuth)
        position = Vector3RotateByQuaternion(Vector3(0, 0, self.distance), rotationAzimuth)
        axis = Vector3Normalize(Vector3CrossProduct(position, Vector3(0, 1, 0)))

        rotationAltitude = QuaternionFromAxisAngle(axis, self.altitude)

        localOffset = Vector3(dt * offsetDeltaX, dt * -offsetDeltaY, 0.0)
        localOffset = Vector3RotateByQuaternion(localOffset, rotationAzimuth)

        self.offset = Vector3Add(self.offset, Vector3RotateByQuaternion(localOffset, rotationAltitude))

        cameraTarget = Vector3Add(self.offset, target)
        eye = Vector3Add(cameraTarget, Vector3RotateByQuaternion(position, rotationAltitude))

        self.cam3d.target = cameraTarget
        self.cam3d.position = eye        

#----------------------------------------------------------------------------------
# Shadow Maps
#----------------------------------------------------------------------------------

class ShadowLight:
    
    def __init__(self):
        
        self.target = Vector3Zero()
        self.position = Vector3Zero()
        self.up = Vector3(0.0, 1.0, 0.0)
        self.target = Vector3Zero()
        self.width = 0
        self.height = 0
        self.near = 0.0
        self.far = 1.0


def LoadShadowMap(width, height):

    target = RenderTexture()
    target.id = rlLoadFramebuffer()
    target.texture.width = width
    target.texture.height = height
    assert target.id != 0
    
    rlEnableFramebuffer(target.id)

    target.depth.id = rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19       #DEPTH_COMPONENT_24BIT?
    target.depth.mipmaps = 1
    rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0)
    assert rlFramebufferComplete(target.id)

    rlDisableFramebuffer()

    return target

def UnloadShadowMap(target):
    
    if target.id > 0:
        rlUnloadFramebuffer(target.id)
        

def BeginShadowMap(target, shadowLight):
    
    BeginTextureMode(target)
    ClearBackground(WHITE)
    
    rlDrawRenderBatchActive()      # Update and draw internal render batch

    rlMatrixMode(RL_PROJECTION)    # Switch to projection matrix
    rlPushMatrix()                 # Save previous matrix, which contains the settings for the 2d ortho projection
    rlLoadIdentity()               # Reset current matrix (projection)

    rlOrtho(
        -shadowLight.width/2, shadowLight.width/2, 
        -shadowLight.height/2, shadowLight.height/2, 
        shadowLight.near, shadowLight.far)

    rlMatrixMode(RL_MODELVIEW)     # Switch back to modelview matrix
    rlLoadIdentity()               # Reset current matrix (modelview)

    # Setup Camera view
    matView = MatrixLookAt(shadowLight.position, shadowLight.target, shadowLight.up)
    rlMultMatrixf(MatrixToFloatV(matView).v)      # Multiply modelview matrix by view matrix (camera)

    rlEnableDepthTest()            # Enable DEPTH_TEST for 3D    


def EndShadowMap():
    rlDrawRenderBatchActive()       # Update and draw internal render batch

    rlMatrixMode(RL_PROJECTION)     # Switch to projection matrix
    rlPopMatrix()                   # Restore previous matrix (projection) from matrix stack

    rlMatrixMode(RL_MODELVIEW)      # Switch back to modelview matrix
    rlLoadIdentity()                # Reset current matrix (modelview)

    rlDisableDepthTest()            # Disable DEPTH_TEST for 2D

    EndTextureMode()

def SetShaderValueShadowMap(shader, locIndex, target):
    if locIndex > -1:
        rlEnableShader(shader.id)
        slotPtr = ffi.new('int*'); slotPtr[0] = 10  # Can be anything 0 to 15, but 0 will probably be taken up
        rlActiveTextureSlot(slotPtr[0])
        rlEnableTexture(target.depth.id)
        rlSetUniform(locIndex, slotPtr, SHADER_UNIFORM_INT, 1)

#----------------------------------------------------------------------------------
# GBuffer
#----------------------------------------------------------------------------------

class GBuffer:
    
    def __init__(self):
        self.id = 0              # OpenGL framebuffer object id
        self.color = Texture()   # Color buffer attachment texture 
        self.normal = Texture()  # Normal buffer attachment texture 
        self.depth = Texture()   # Depth buffer attachment texture


def LoadGBuffer(width, height):
    
    target = GBuffer()
    target.id = rlLoadFramebuffer()
    assert target.id
    
    rlEnableFramebuffer(target.id)

    target.color.id = rlLoadTexture(ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, 1)
    target.color.width = width
    target.color.height = height
    target.color.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    target.color.mipmaps = 1
    rlFramebufferAttach(target.id, target.color.id, RL_ATTACHMENT_COLOR_CHANNEL0, RL_ATTACHMENT_TEXTURE2D, 0)
    
    target.normal.id = rlLoadTexture(ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R16G16B16A16, 1)
    target.normal.width = width
    target.normal.height = height
    target.normal.format = PIXELFORMAT_UNCOMPRESSED_R16G16B16A16
    target.normal.mipmaps = 1
    rlFramebufferAttach(target.id, target.normal.id, RL_ATTACHMENT_COLOR_CHANNEL1, RL_ATTACHMENT_TEXTURE2D, 0)
    
    target.depth.id = rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19       #DEPTH_COMPONENT_24BIT?
    target.depth.mipmaps = 1
    rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0)

    assert rlFramebufferComplete(target.id)

    rlDisableFramebuffer()

    return target


def UnloadGBuffer(target):

    if target.id > 0:
        rlUnloadFramebuffer(target.id)


def BeginGBuffer(target, camera):
    
    rlDrawRenderBatchActive()       # Update and draw internal render batch

    rlEnableFramebuffer(target.id)  # Enable render target
    rlActiveDrawBuffers(2) 

    # Set viewport and RLGL internal framebuffer size
    rlViewport(0, 0, target.color.width, target.color.height)
    rlSetFramebufferWidth(target.color.width)
    rlSetFramebufferHeight(target.color.height)

    ClearBackground(BLACK)

    rlMatrixMode(RL_PROJECTION)    # Switch to projection matrix
    rlPushMatrix()                 # Save previous matrix, which contains the settings for the 2d ortho projection
    rlLoadIdentity()               # Reset current matrix (projection)

    aspect = float(target.color.width)/float(target.color.height)

    # NOTE: zNear and zFar values are important when computing depth buffer values
    if camera.projection == CAMERA_PERSPECTIVE:

        # Setup perspective projection
        top = rlGetCullDistanceNear()*np.tan(camera.fovy*0.5*DEG2RAD)
        right = top*aspect

        rlFrustum(-right, right, -top, top, rlGetCullDistanceNear(), rlGetCullDistanceFar())

    elif camera.projection == CAMERA_ORTHOGRAPHIC:

        # Setup orthographic projection
        top = camera.fovy/2.0
        right = top*aspect

        rlOrtho(-right, right, -top,top, rlGetCullDistanceNear(), rlGetCullDistanceFar())

    rlMatrixMode(RL_MODELVIEW)     # Switch back to modelview matrix
    rlLoadIdentity()               # Reset current matrix (modelview)

    # Setup Camera view
    matView = MatrixLookAt(camera.position, camera.target, camera.up)
    rlMultMatrixf(MatrixToFloatV(matView).v)      # Multiply modelview matrix by view matrix (camera)

    rlEnableDepthTest()            # Enable DEPTH_TEST for 3D


def EndGBuffer(windowWidth, windowHeight):
    
    rlDrawRenderBatchActive()       # Update and draw internal render batch
    
    rlDisableDepthTest()            # Disable DEPTH_TEST for 2D
    rlActiveDrawBuffers(1) 
    rlDisableFramebuffer()          # Disable render target (fbo)

    rlMatrixMode(RL_PROJECTION)         # Switch to projection matrix
    rlPopMatrix()                   # Restore previous matrix (projection) from matrix stack
    rlLoadIdentity()                    # Reset current matrix (projection)
    rlOrtho(0, windowWidth, windowHeight, 0, 0.0, 1.0)

    rlMatrixMode(RL_MODELVIEW)          # Switch back to modelview matrix
    rlLoadIdentity()                    # Reset current matrix (modelview)


#----------------------------------------------------------------------------------
# Geno Character and Animation
#----------------------------------------------------------------------------------

def FileRead(out, size, f):
    ffi.memmove(out, f.read(size), size)

def LoadGenoModel(fileName):

    model = Model()
    model.transform = MatrixIdentity()
  
    with open(fileName, "rb") as f:
        
        model.materialCount = 1
        model.materials = MemAlloc(model.materialCount * ffi.sizeof(Mesh()))
        model.materials[0] = LoadMaterialDefault()

        model.meshCount = 1
        model.meshMaterial = MemAlloc(model.meshCount * ffi.sizeof(Mesh()))
        model.meshMaterial[0] = 0

        model.meshes = MemAlloc(model.meshCount * ffi.sizeof(Mesh()))
        model.meshes[0].vertexCount = struct.unpack('I', f.read(4))[0]
        model.meshes[0].triangleCount = struct.unpack('I', f.read(4))[0]
        model.boneCount = struct.unpack('I', f.read(4))[0]

        model.meshes[0].boneCount = model.boneCount
        model.meshes[0].vertices = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.meshes[0].texcoords = MemAlloc(model.meshes[0].vertexCount * 2 * ffi.sizeof("float"))
        model.meshes[0].normals = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.meshes[0].boneIds = MemAlloc(model.meshes[0].vertexCount * 4 * ffi.sizeof("unsigned char"))
        model.meshes[0].boneWeights = MemAlloc(model.meshes[0].vertexCount * 4 * ffi.sizeof("float"))
        model.meshes[0].indices = MemAlloc(model.meshes[0].triangleCount * 3 * ffi.sizeof("unsigned short"))
        model.meshes[0].animVertices = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.meshes[0].animNormals = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.bones =  MemAlloc(model.boneCount * ffi.sizeof(BoneInfo()))
        model.bindPose =  MemAlloc(model.boneCount * ffi.sizeof(Transform()))
        
        FileRead(model.meshes[0].vertices, ffi.sizeof("float") * model.meshes[0].vertexCount * 3, f)
        FileRead(model.meshes[0].texcoords, ffi.sizeof("float") * model.meshes[0].vertexCount * 2, f)
        FileRead(model.meshes[0].normals, ffi.sizeof("float") * model.meshes[0].vertexCount * 3, f)
        FileRead(model.meshes[0].boneIds, ffi.sizeof("unsigned char") * model.meshes[0].vertexCount * 4, f)
        FileRead(model.meshes[0].boneWeights, ffi.sizeof("float") * model.meshes[0].vertexCount * 4, f)
        FileRead(model.meshes[0].indices, ffi.sizeof("unsigned short") * model.meshes[0].triangleCount * 3, f)
        ffi.memmove(model.meshes[0].animVertices, model.meshes[0].vertices, ffi.sizeof("float") * model.meshes[0].vertexCount * 3)
        ffi.memmove(model.meshes[0].animNormals, model.meshes[0].normals, ffi.sizeof("float") * model.meshes[0].vertexCount * 3)
        FileRead(model.bones, ffi.sizeof(BoneInfo()) * model.boneCount, f)
        FileRead(model.bindPose, ffi.sizeof(Transform()) * model.boneCount, f)
        
        model.meshes[0].boneMatrices = MemAlloc(model.boneCount * ffi.sizeof(Matrix()))
        for i in range(model.boneCount):
            model.meshes[0].boneMatrices[i] = MatrixIdentity()
    
    UploadMesh(ffi.addressof(model.meshes[0]), True)
    
    return model


def GetModelBindPoseAsNumpyArrays(model):
    
    bindPos = np.zeros([model.boneCount, 3])
    bindRot = np.zeros([model.boneCount, 4])
    
    for boneId in range(model.boneCount):
        bindTransform = model.bindPose[boneId]
        bindPos[boneId] = (bindTransform.translation.x, bindTransform.translation.y, bindTransform.translation.z)
        bindRot[boneId] = (bindTransform.rotation.w, bindTransform.rotation.x, bindTransform.rotation.y, bindTransform.rotation.z)
        
    return bindPos, bindRot
    
def UpdateModelPoseFromNumpyArrays(model, bindPos, bindRot, animPos, animRot):
    
    meshPos = quat.mul_vec(animRot, quat.inv_mul_vec(bindRot, -bindPos)) + animPos
    meshRot = quat.mul_inv(animRot, bindRot)
    
    matArray = np.frombuffer(ffi.buffer(
        model.meshes[0].boneMatrices, model.boneCount * 4 * 4 * 4), 
        dtype=np.float32).reshape([model.boneCount, 4, 4])
    
    matArray[:,:3,:3] = quat.to_xform(meshRot)
    matArray[:,:3,3] = meshPos


#----------------------------------------------------------------------------------
# Debug Draw
#----------------------------------------------------------------------------------

def DrawTransform(position, rotation, scale):
    
    rotation = rotation[...,np.array([1,2,3,0])]
    rotMatrix = QuaternionToMatrix(Vector4(*rotation))
  
    DrawLine3D(
        Vector3(*position),
        Vector3Add(Vector3(*position), Vector3(scale * rotMatrix.m0, scale * rotMatrix.m1, scale * rotMatrix.m2)),
        RED)
        
    DrawLine3D(
        Vector3(*position),
        Vector3Add(Vector3(*position), Vector3(scale * rotMatrix.m4, scale * rotMatrix.m5, scale * rotMatrix.m6)),
        GREEN)
        
    DrawLine3D(
        Vector3(*position),
        Vector3Add(Vector3(*position), Vector3(scale * rotMatrix.m8, scale * rotMatrix.m9, scale * rotMatrix.m10)),
        BLUE)

def DrawSkeleton(positions, rotations, parents, color):
    
    for i in range(len(positions)):
    
        DrawSphereWires(
            Vector3(*positions[i]),
            0.01,
            4,
            6,
            color)

        DrawTransform(positions[i], rotations[i], 0.1)

        if parents[i] != -1:
        
            DrawLine3D(
                Vector3(*positions[i]),
                Vector3(*positions[parents[i]]),
                color)

def DrawTrajectory(Tpos, Tdir, color):
    
    for i in range(len(Tpos)):
        DrawSphere(Vector3(*Tpos[i]), 0.05, color)
        DrawCapsule(Vector3(*Tpos[i]), Vector3(*(Tpos[i] + 0.25 * Tdir[i])), 0.01, 5, 7, color)

def DrawFeatures(X, Xoffset, Xscale, rootPos, rootRot, color):
    
    assert len(X) == 27
    
    X = X * Xscale + Xoffset
    
    p0, p1 = X[0:3], X[3:6]
    v0, v1, v2 = X[6:9], X[9:12], X[12:15]
    tp0, tp1, tp2 = X[15:17], X[17:19], X[19:21]
    td0, td1, td2 = X[21:23], X[23:25], X[25:27]
    
    p0 = quat.mul_vec(rootRot, p0) + rootPos
    p1 = quat.mul_vec(rootRot, p1) + rootPos
    v0 = quat.mul_vec(rootRot, v0)
    v1 = quat.mul_vec(rootRot, v1)
    
    tp0 = quat.mul_vec(rootRot, np.array([tp0[0], 0.0, tp0[1]])) + rootPos
    tp1 = quat.mul_vec(rootRot, np.array([tp1[0], 0.0, tp1[1]])) + rootPos
    tp2 = quat.mul_vec(rootRot, np.array([tp2[0], 0.0, tp2[1]])) + rootPos
    
    td0 = quat.mul_vec(rootRot, np.array([td0[0], 0.0, td0[1]]))
    td1 = quat.mul_vec(rootRot, np.array([td1[0], 0.0, td1[1]]))
    td2 = quat.mul_vec(rootRot, np.array([td2[0], 0.0, td2[1]]))
    
    DrawSphere(Vector3(*p0), 0.05, color)
    DrawSphere(Vector3(*p1), 0.05, color)
    DrawCapsule(Vector3(*p0), Vector3(*(p0 + v0)), 0.01, 5, 7, color)
    DrawCapsule(Vector3(*p1), Vector3(*(p1 + v1)), 0.01, 5, 7, color)

    DrawSphere(Vector3(*tp0), 0.05, color)
    DrawSphere(Vector3(*tp1), 0.05, color)
    DrawSphere(Vector3(*tp2), 0.05, color)
    DrawCapsule(Vector3(*tp0), Vector3(*(tp0 + 0.25 * td0)), 0.01, 5, 7, color)
    DrawCapsule(Vector3(*tp1), Vector3(*(tp1 + 0.25 * td1)), 0.01, 5, 7, color)
    DrawCapsule(Vector3(*tp2), Vector3(*(tp2 + 0.25 * td2)), 0.01, 5, 7, color)


#----------------------------------------------------------------------------------
# Motion Matching
#----------------------------------------------------------------------------------

def LoadDatabase():
    
    import scipy.signal as signal
    
    files = [
        ('resources/pushAndStumble1_subject5.bvh', 397,  706), 
        # Running
        ('resources/run1_subject5.bvh',             172, 14136),
        # Walking
        ('resources/walk1_subject5.bvh',            160, 15518),
    ]
    
    Ypos = []
    Yrot = []
    Yvel = []
    Yang = []
    YrangeStarts = []
    YrangeStops = []
    
    for filename, start, stop in files:
    
        for mirrored in [True, False]:
        
            bvhData = bvh.load(filename)
            
            pos = 0.01 * bvhData['positions'][start:stop].copy().astype(np.float32)
            rot = quat.unroll(quat.from_euler(np.radians(bvhData['rotations'][start:stop]), order=bvhData['order']))
            
            # First compute world space positions/rotations
            gloRot, gloPos = quat.fk(rot, pos, bvhData['parents'])
            
            if mirrored:
                
                mirror_bones = []
                for ni, n in enumerate(bvhData['names']):
                    if 'Right' in n and n.replace('Right', 'Left') in bvhData['names']:
                        mirror_bones.append(bvhData['names'].index(n.replace('Right', 'Left')))
                    elif 'Left' in n and n.replace('Left', 'Right') in bvhData['names']:
                        mirror_bones.append(bvhData['names'].index(n.replace('Left', 'Right')))
                    else:
                        mirror_bones.append(ni)
                
                mirror_bones = np.array(mirror_bones)
                
                gloRot, gloPos = quat.fk(rot, pos, bvhData['parents'])
                gloPos = np.array([-1, 1, 1]) * gloPos[:,mirror_bones]
                gloRot = np.array([1, 1, -1, -1]) * gloRot[:,mirror_bones]
                rot, pos = quat.ik(gloRot, gloPos, bvhData['parents'])
            
            # Specify joints to use for simulation bone 
            simPosJoint = bvhData['names'].index("Spine2")
            simRotJoint = bvhData['names'].index("Hips")
            
            # Position comes from spine joint
            simPos = np.array([1.0, 0.0, 1.0]) * gloPos[:,simPosJoint:simPosJoint+1]
            simPos = signal.savgol_filter(simPos, 31, 3, axis=0, mode='interp')
            
            # Direction comes from projected hip forward direction
            simDir = np.array([1.0, 0.0, 1.0]) * quat.mul_vec(gloRot[:,simRotJoint:simRotJoint+1], np.array([0.0, 0.0, 1.0]))

            # We need to re-normalize the direction after both projection and smoothing
            simDir = simDir / np.sqrt(np.sum(np.square(simDir), axis=-1))[...,np.newaxis]
            simDir = signal.savgol_filter(simDir, 61, 3, axis=0, mode='interp')
            simDir = simDir / np.sqrt(np.sum(np.square(simDir), axis=-1)[...,np.newaxis])
            
            # Extract rotation from direction
            simRot = quat.normalize(quat.between(np.array([0, 0, 1]), simDir))
            
            # Transform first joints to be local to sim and append sim as root bone
            pos[:,0:1] = quat.mul_vec(quat.inv(simRot), pos[:,0:1] - simPos)
            rot[:,0:1] = quat.mul(quat.inv(simRot), rot[:,0:1])
            
            pos = np.concatenate([simPos, pos], axis=1)
            rot = np.concatenate([simRot, rot], axis=1)
            
            parents = np.concatenate([[-1], bvhData['parents'] + 1])
            names = np.array(['Simulation'] + bvhData['names'])
            
            # Compute velocities via central difference
            vel = np.empty_like(pos)
            vel[1:-1] = (
                0.5 * (pos[2:  ] - pos[1:-1]) * 60.0 +
                0.5 * (pos[1:-1] - pos[ :-2]) * 60.0)
            vel[ 0] = vel[ 1] - (vel[ 3] - vel[ 2])
            vel[-1] = vel[-2] + (vel[-2] - vel[-3])
            
            # Same for angular velocities
            ang = np.zeros_like(pos)
            ang[1:-1] = (
                0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rot[2:  ], rot[1:-1]))) * 60.0 +
                0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rot[1:-1], rot[ :-2]))) * 60.0)
            ang[ 0] = ang[ 1] - (ang[ 3] - ang[ 2])
            ang[-1] = ang[-2] + (ang[-2] - ang[-3])
            
            # Append to Database

            Ypos.append(pos)
            Yvel.append(vel)
            Yrot.append(rot)
            Yang.append(ang)
            
            offset = 0 if len(YrangeStarts) == 0 else YrangeStops[-1] 

            YrangeStarts.append(offset)
            YrangeStops.append(offset + len(pos))
        
    Ypos = np.concatenate(Ypos, axis=0)
    Yrot = np.concatenate(Yrot, axis=0)
    Yvel = np.concatenate(Yvel, axis=0)
    Yang = np.concatenate(Yang, axis=0)
    
    YrangeStarts = np.array(YrangeStarts)
    YrangeStops = np.array(YrangeStops)
    
    return dict(
        Ypos=Ypos,
        Yrot=Yrot, 
        Yvel=Yvel, 
        Yang=Yang, 
        YrangeStarts=YrangeStarts, 
        YrangeStops=YrangeStops, 
        parents=parents, 
        names=names)
    
    
def ComputeDatabaseFeatures(Y):
       
    Ypos = Y['Ypos']
    Yrot = Y['Yrot']
    Yvel = Y['Yvel']
    Yang = Y['Yang']
    YrangeStarts = Y['YrangeStarts']
    YrangeStops = Y['YrangeStops']
    parents = Y['parents']
    names = Y['names'].tolist()
    
    posJoints = np.array([names.index(n) for n in ['LeftToeBase', 'RightToeBase']])
    velJoints = np.array([names.index(n) for n in ['LeftToeBase', 'RightToeBase', 'Hips']])
    
    YgloRot, YgloPos, YgloAng, YgloVel = quat.fk_vel(Yrot, Ypos, Yang, Yvel, parents)
    YrootDir = quat.mul_vec(YgloRot[:,0], np.array([0,0,1]))
    
    Xpos = quat.inv_mul_vec(YgloRot[:,0:1], YgloPos[:,posJoints] - YgloPos[:,0:1]).reshape([len(Ypos), 6])
    Xvel = quat.inv_mul_vec(YgloRot[:,0:1], YgloVel[:,velJoints]).reshape([len(Ypos), 9])
    
    XtrajPos = np.zeros([len(Ypos), 6])
    XtrajDir = np.zeros([len(Ypos), 6])
    for rs, re in zip(YrangeStarts, YrangeStops):        
        ft0 = np.clip(np.arange(rs, re) + 20, rs, re - 1)
        ft1 = np.clip(np.arange(rs, re) + 40, rs, re - 1)
        ft2 = np.clip(np.arange(rs, re) + 60, rs, re - 1)
        
        XtrajPos[rs:re,0:2] = quat.inv_mul_vec(YgloRot[rs:re,0], YgloPos[ft0,0] - YgloPos[rs:re,0])[:,np.array([0,2])]
        XtrajPos[rs:re,2:4] = quat.inv_mul_vec(YgloRot[rs:re,0], YgloPos[ft1,0] - YgloPos[rs:re,0])[:,np.array([0,2])]
        XtrajPos[rs:re,4:6] = quat.inv_mul_vec(YgloRot[rs:re,0], YgloPos[ft2,0] - YgloPos[rs:re,0])[:,np.array([0,2])]
        
        XtrajDir[rs:re,0:2] = quat.inv_mul_vec(YgloRot[rs:re,0], YrootDir[ft0])[:,np.array([0,2])]
        XtrajDir[rs:re,2:4] = quat.inv_mul_vec(YgloRot[rs:re,0], YrootDir[ft1])[:,np.array([0,2])]
        XtrajDir[rs:re,4:6] = quat.inv_mul_vec(YgloRot[rs:re,0], YrootDir[ft2])[:,np.array([0,2])]
    
    X = np.concatenate([Xpos, Xvel, XtrajPos, XtrajDir], axis=-1)
    
    Xoffset = X.mean(axis=0)
    
    Xscale = np.concatenate([
        Xpos.std(axis=0).mean().repeat(Xpos.shape[1]),
        Xvel.std(axis=0).mean().repeat(Xvel.shape[1]),
        XtrajPos.std(axis=0).mean().repeat(XtrajPos.shape[1]),
        XtrajDir.std(axis=0).mean().repeat(XtrajDir.shape[1]),
    ], axis=-1)
    
    X = (X - Xoffset) / Xscale
    
    return X, Xoffset, Xscale
    
    
def ComputeRuntimeFeatures(X, i, Xoffset, Xscale, Tpos, Tdir, rootPos, rootRot):
    
    Xpos = X[i,0:6] * Xscale[0:6] + Xoffset[0:6]
    Xvel = X[i,6:15] * Xscale[6:15] + Xoffset[6:15]
    
    XtrajPos = quat.inv_mul_vec(rootRot, Tpos - rootPos)[:,np.array([0,2])].ravel()
    XtrajDir = quat.inv_mul_vec(rootRot, Tdir)[:,np.array([0,2])].ravel()
    
    return (np.concatenate([Xpos, Xvel, XtrajPos, XtrajDir], axis=-1) - Xoffset) / Xscale
    
    
def HalflifeToDamping(halflife, eps = 1e-5):
    return (4.0 * 0.69314718056) / (halflife + eps)


def TrajectorySpringPosition(pos, vel, acc, desiredVel, halflife, dt):
    
    y = HalflifeToDamping(halflife) / 2.0	
    j0 = vel - desiredVel
    j1 = acc + j0*y
    eydt = np.exp(-y*dt)

    return (
        eydt*(((-j1)/(y*y)) + ((-j0 - j1*dt)/y)) + 
        (j1/(y*y)) + j0/y + desiredVel * dt + pos,
        eydt*(j0 + j1*dt) + desiredVel,
        eydt*(acc - j1*y*dt))


def TrajectorySpringRotation(rot, ang, desiredRot, halflife, dt):
    
    y = HalflifeToDamping(halflife) / 2.0
	
    j0 = quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rot, desiredRot)))
    j1 = ang + j0*y
	
    eydt = np.exp(-y*dt)

    return (
        quat.mul(quat.from_scaled_angle_axis(eydt*(j0 + j1*dt)), desiredRot),
        eydt*(ang - j1*y*dt))


def GamepadStick(stick, deadzone=0.2):
    
    gamepadx = GetGamepadAxisMovement(0, GAMEPAD_AXIS_LEFT_X if stick == 'left' else GAMEPAD_AXIS_RIGHT_X)
    gamepady = GetGamepadAxisMovement(0, GAMEPAD_AXIS_LEFT_Y if stick == 'left' else GAMEPAD_AXIS_RIGHT_Y)
    gamepadmag = np.sqrt(gamepadx*gamepadx + gamepady*gamepady)
    
    if gamepadmag > deadzone:
        gamepaddirx = gamepadx / gamepadmag
        gamepaddiry = gamepady / gamepadmag
        gamepadclippedmag = 1.0 if gamepadmag > 1.0 else gamepadmag*gamepadmag
        gamepadx = gamepaddirx * gamepadclippedmag
        gamepady = gamepaddiry * gamepadclippedmag
    else:
        gamepadx = 0.0
        gamepady = 0.0
    
    return np.array([gamepadx, 0.0, gamepady])


def DecaySpringDamperPosition(x, v, halflife, dt):
    
    y = HalflifeToDamping(halflife) / 2.0 
    j1 = v + x*y
    eydt = np.exp(-y*dt)

    return (
        eydt*(x + j1*dt),
        eydt*(v - j1*y*dt))

def DecaySpringDamperRotation(x, v, halflife, dt):
    
    y = HalflifeToDamping(halflife) / 2.0
    j0 = quat.to_scaled_angle_axis(x)
    j1 = v + j0*y
    eydt = np.exp(-y*dt)

    return (
        quat.from_scaled_angle_axis(eydt*(j0 + j1*dt)),
        eydt*(v - j1*y*dt))

def InertializeTransitionPosition(offPos, offVel, srcPos, srcVel, dstPos, dstVel):
    return ((offPos + srcPos) - dstPos, ((offVel + srcVel) -  dstVel))
    
def InertializeTransitionRotation(offRot, offAng, srcRot, srcAng, dstRot, dstAng):
    return (
        quat.abs(quat.mul_inv(quat.mul(offRot, srcRot), dstRot)),
        ((offAng + srcAng) -  dstAng))
        
def InertializeUpdatePosition(offPos, offVel, inPos, inVel, halflife, deltatime):
    offPos, offVel = DecaySpringDamperPosition(offPos, offVel, halflife, deltatime)
    return (offPos, offVel, inPos + offPos, inVel + offVel)

def InertializeUpdateRotation(offRot, offAng, inRot, inAng, halflife, deltatime):
    offRot, offAng = DecaySpringDamperRotation(offRot, offAng, halflife, deltatime)    
    return (offRot, offAng, quat.mul(offRot, inRot), inAng + offAng)

#----------------------------------------------------------------------------------
# App
#----------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Init Window
    
    screenWidth = 1280
    screenHeight = 720
    
    SetConfigFlags(FLAG_VSYNC_HINT)
    InitWindow(screenWidth, screenHeight, b"GenoViewPython")
    SetTargetFPS(30)

    # Shaders
    
    shadowShader = LoadShader(b"./resources/shadow.vs", b"./resources/shadow.fs")
    shadowShaderLightClipNear = GetShaderLocation(shadowShader, b"lightClipNear")
    shadowShaderLightClipFar = GetShaderLocation(shadowShader, b"lightClipFar")
    
    skinnedShadowShader = LoadShader(b"./resources/skinnedShadow.vs", b"./resources/shadow.fs")
    skinnedShadowShaderLightClipNear = GetShaderLocation(skinnedShadowShader, b"lightClipNear")
    skinnedShadowShaderLightClipFar = GetShaderLocation(skinnedShadowShader, b"lightClipFar")
    
    skinnedBasicShader = LoadShader(b"./resources/skinnedBasic.vs", b"./resources/basic.fs")
    skinnedBasicShaderSpecularity = GetShaderLocation(skinnedBasicShader, b"specularity")
    skinnedBasicShaderGlossiness = GetShaderLocation(skinnedBasicShader, b"glossiness")
    skinnedBasicShaderCamClipNear = GetShaderLocation(skinnedBasicShader, b"camClipNear")
    skinnedBasicShaderCamClipFar = GetShaderLocation(skinnedBasicShader, b"camClipFar")

    basicShader = LoadShader(b"./resources/basic.vs", b"./resources/basic.fs")
    basicShaderSpecularity = GetShaderLocation(basicShader, b"specularity")
    basicShaderGlossiness = GetShaderLocation(basicShader, b"glossiness")
    basicShaderCamClipNear = GetShaderLocation(basicShader, b"camClipNear")
    basicShaderCamClipFar = GetShaderLocation(basicShader, b"camClipFar")
    
    lightingShader = LoadShader(b"./resources/post.vs", b"./resources/lighting.fs")
    lightingShaderGBufferColor = GetShaderLocation(lightingShader, b"gbufferColor")
    lightingShaderGBufferNormal = GetShaderLocation(lightingShader, b"gbufferNormal")
    lightingShaderGBufferDepth = GetShaderLocation(lightingShader, b"gbufferDepth")
    lightingShaderSSAO = GetShaderLocation(lightingShader, b"ssao")
    lightingShaderCamPos = GetShaderLocation(lightingShader, b"camPos")
    lightingShaderCamInvViewProj = GetShaderLocation(lightingShader, b"camInvViewProj")
    lightingShaderLightDir = GetShaderLocation(lightingShader, b"lightDir")
    lightingShaderSunColor = GetShaderLocation(lightingShader, b"sunColor")
    lightingShaderSunStrength = GetShaderLocation(lightingShader, b"sunStrength")
    lightingShaderSkyColor = GetShaderLocation(lightingShader, b"skyColor")
    lightingShaderSkyStrength = GetShaderLocation(lightingShader, b"skyStrength")
    lightingShaderGroundStrength = GetShaderLocation(lightingShader, b"groundStrength")
    lightingShaderAmbientStrength = GetShaderLocation(lightingShader, b"ambientStrength")
    lightingShaderExposure = GetShaderLocation(lightingShader, b"exposure")
    lightingShaderCamClipNear = GetShaderLocation(lightingShader, b"camClipNear")
    lightingShaderCamClipFar = GetShaderLocation(lightingShader, b"camClipFar")
    
    ssaoShader = LoadShader(b"./resources/post.vs", b"./resources/ssao.fs")
    ssaoShaderGBufferNormal = GetShaderLocation(ssaoShader, b"gbufferNormal")
    ssaoShaderGBufferDepth = GetShaderLocation(ssaoShader, b"gbufferDepth")
    ssaoShaderCamView = GetShaderLocation(ssaoShader, b"camView")
    ssaoShaderCamProj = GetShaderLocation(ssaoShader, b"camProj")
    ssaoShaderCamInvProj = GetShaderLocation(ssaoShader, b"camInvProj")
    ssaoShaderCamInvViewProj = GetShaderLocation(ssaoShader, b"camInvViewProj")
    ssaoShaderLightViewProj = GetShaderLocation(ssaoShader, b"lightViewProj")
    ssaoShaderShadowMap = GetShaderLocation(ssaoShader, b"shadowMap")
    ssaoShaderShadowInvResolution = GetShaderLocation(ssaoShader, b"shadowInvResolution")
    ssaoShaderCamClipNear = GetShaderLocation(ssaoShader, b"camClipNear")
    ssaoShaderCamClipFar = GetShaderLocation(ssaoShader, b"camClipFar")
    ssaoShaderLightClipNear = GetShaderLocation(ssaoShader, b"lightClipNear")
    ssaoShaderLightClipFar = GetShaderLocation(ssaoShader, b"lightClipFar")
    ssaoShaderLightDir = GetShaderLocation(ssaoShader, b"lightDir")
    
    blurShader = LoadShader(b"./resources/post.vs", b"./resources/blur.fs")
    blurShaderGBufferNormal = GetShaderLocation(blurShader, b"gbufferNormal")
    blurShaderGBufferDepth = GetShaderLocation(blurShader, b"gbufferDepth")
    blurShaderInputTexture = GetShaderLocation(blurShader, b"inputTexture")
    blurShaderCamInvProj = GetShaderLocation(blurShader, b"camInvProj")
    blurShaderCamClipNear = GetShaderLocation(blurShader, b"camClipNear")
    blurShaderCamClipFar = GetShaderLocation(blurShader, b"camClipFar")
    blurShaderInvTextureResolution = GetShaderLocation(blurShader, b"invTextureResolution")
    blurShaderBlurDirection = GetShaderLocation(blurShader, b"blurDirection")

    fxaaShader = LoadShader(b"./resources/post.vs", b"./resources/fxaa.fs")
    fxaaShaderInputTexture = GetShaderLocation(fxaaShader, b"inputTexture")
    fxaaShaderInvTextureResolution = GetShaderLocation(fxaaShader, b"invTextureResolution")

    # Uniform Array Parameters

    lightClipNearPtr = ffi.new("float*")
    lightClipFarPtr = ffi.new("float*")
    camClipNearPtr = ffi.new("float*")
    camClipFarPtr = ffi.new("float*")
    specularityPtr = ffi.new('float*')
    glossinessPtr = ffi.new('float*')
    sunStrengthPtr = ffi.new('float*')
    skyStrengthPtr = ffi.new('float*')
    groundStrengthPtr = ffi.new('float*')
    ambientStrengthPtr = ffi.new('float*')
    exposurePtr = ffi.new('float*')

    # Objects
    
    groundMesh = GenMeshPlane(20.0, 20.0, 10, 10)
    groundModel = LoadModelFromMesh(groundMesh)
    groundPosition = Vector3(0.0, -0.01, 0.0)
    
    genoModel = LoadGenoModel(b"./resources/Geno.bin")
    genoPosition = Vector3(0.0, 0.0, 0.0)
    
    bindPos, bindRot = GetModelBindPoseAsNumpyArrays(genoModel)
    
    # Camera
    
    camera = Camera()
    
    rlSetClipPlanes(0.01, 50.0)
    
    # Shadows
    
    lightDir = Vector3Normalize(Vector3(0.35, -1.0, -0.35))
    
    shadowLight = ShadowLight()
    shadowLight.target = Vector3Zero()
    shadowLight.position = Vector3Scale(lightDir, -5.0)
    shadowLight.up = Vector3(0.0, 1.0, 0.0)
    shadowLight.width = 5.0
    shadowLight.height = 5.0
    shadowLight.near = 0.01
    shadowLight.far = 10.0
    
    shadowWidth = 1024
    shadowHeight = 1024
    shadowInvResolution = Vector2(1.0 / shadowWidth, 1.0 / shadowHeight)
    shadowMap = LoadShadowMap(shadowWidth, shadowHeight)    
    
    # GBuffer and Render Textures
    
    gbuffer = LoadGBuffer(screenWidth, screenHeight)
    lighted = LoadRenderTexture(screenWidth, screenHeight)
    ssaoFront = LoadRenderTexture(screenWidth, screenHeight)
    ssaoBack = LoadRenderTexture(screenWidth, screenHeight)
    
    # Create Pose Database
    
    np.set_printoptions(suppress=True)
    flt_max = np.finfo(np.float32).max
    
    if not os.path.exists('Y.npz'):
        Y = LoadDatabase()
        np.savez('Y.npz', **Y)
    else:
        Y = dict(np.load('Y.npz'))
    
    Ypos = Y['Ypos']
    Yrot = Y['Yrot']
    Yvel = Y['Yvel']
    Yang = Y['Yang']
    YrangeStarts = Y['YrangeStarts']
    YrangeStops = Y['YrangeStops']
    parents = Y['parents']
    names = Y['names']
    
    # Create matching feature database
    
    if not os.path.exists('X.npz'):
        X, Xoffset, Xscale = ComputeDatabaseFeatures(Y)
        np.savez('X.npz', X=X, Xoffset=Xoffset, Xscale=Xscale)
    else:
        Xdata = dict(np.load('X.npz'))
        X = Xdata['X']
        Xoffset = Xdata['Xoffset']
        Xscale = Xdata['Xscale']
    
    # Fit KD trees
    
    from scipy.spatial import cKDTree
    Xtrees = []
    for rs, re in zip(YrangeStarts, YrangeStops):
        Xtrees.append(cKDTree(X[rs:re-60])) # Trim last second from end
    
    # Root
    
    rootPos = np.array([0.0, 0.0, 0.0])
    rootVel = np.array([0.0, 0.0, 0.0])
    rootAcc = np.array([0.0, 0.0, 0.0])
    rootRot = np.array([1.0, 0.0, 0.0, 0.0])
    rootAng = np.array([0.0, 0.0, 0.0])
    
    desiredVel = np.array([0.0, 0.0, 0.0])
    desiredDir = np.array([0.0, 0.0, 1.0])
    
    animRange = 0
    animFrame = 0
    searchTime = 0.15
    searchTimer = searchTime
    
    offPos = np.zeros([len(parents) - 1, 3])
    offRot = quat.eye([len(parents) - 1])
    offVel = np.zeros([len(parents) - 1, 3])
    offAng = np.zeros([len(parents) - 1, 3])
    
    # Go
    
    while not WindowShouldClose():
        
        deltaTime = 1.0 / 30.0
    
        # Predict Trajectory
    
        leftStick, rightStick = GamepadStick('left'), GamepadStick('right')
        
        desiredVel = 5.0 * leftStick
        
        leftStickMag = np.sqrt(np.sum(leftStick*leftStick))
        rightStickMag = np.sqrt(np.sum(rightStick*rightStick))
        
        if rightStickMag > 0.01:
            desiredDir = rightStick / rightStickMag
        elif leftStickMag > 0.01:
            desiredDir = leftStick / leftStickMag
        
        desiredRot = quat.normalize(quat.between(np.array([0,0,1]), desiredDir))
        
        velHalflife = 0.2
        rotHalflife = 0.2
        
        Ttimes = np.array([20, 40, 60]) / 60.0
        
        Tpos, _, _ = TrajectorySpringPosition(
            rootPos, rootVel, rootAcc, desiredVel, velHalflife, Ttimes[...,None])
        
        Trot, _ = TrajectorySpringRotation(
            rootRot, rootAng, desiredRot, rotHalflife, Ttimes[...,None])
        
        Tdir = quat.mul_vec(Trot, np.array([0,0,1]))
        
        # Search
    
        if searchTimer <= 0.0:    
            
            Xquery = ComputeRuntimeFeatures(X, animFrame, Xoffset, Xscale, Tpos, Tdir, rootPos, rootRot)

            bestRange = animRange
            bestFrame = animFrame
            currentBias = 0.01
            approxBias = 0.01
            
            if bestFrame < YrangeStops[bestRange] - searchTime:                
                best = np.sqrt(np.sum(np.square(Xquery - X[bestFrame]), axis=-1)) - currentBias
            else:
                best = flt_max
            
            for ri, (tree, rs, re) in enumerate(zip(Xtrees, YrangeStarts, YrangeStops)):
                dist, k = tree.query(Xquery, eps=approxBias, distance_upper_bound=best)
                if dist < best:
                    best = dist
                    bestRange = ri
                    bestFrame = rs + k            
            
            if bestRange != animRange or bestFrame != animFrame:
                
                # Transition
                
                offPos, offVel = InertializeTransitionPosition(
                    offPos, offVel,
                    Ypos[animFrame,1:], Yvel[animFrame,1:],
                    Ypos[bestFrame,1:], Yvel[bestFrame,1:])
                    
                offRot, offAng = InertializeTransitionRotation(
                    offRot, offAng,
                    Yrot[animFrame,1:], Yang[animFrame,1:],
                    Yrot[bestFrame,1:], Yang[bestFrame,1:])
                
                animRange = bestRange
                animFrame = bestFrame
            
            searchTimer = searchTime
            
        # Advance animation (Database is 60fps, framerate is 30fps so we add 2)
        
        animFrame = np.clip(animFrame + 2, YrangeStarts[animRange], YrangeStops[animRange] - 1)
        searchTimer -= deltaTime
        
        if animFrame >= YrangeStops[animRange] - 4: # End of range so force search next frame
            searchTimer = 0.0
        
        # Update Root (not inertializing root just for code simplicity)
        
        _, _, rootAcc = TrajectorySpringPosition(rootPos, rootVel, rootAcc, desiredVel, rotHalflife, deltaTime)
        rootVel = quat.mul_vec(rootRot, quat.inv_mul_vec(Yrot[animFrame,0], Yvel[animFrame,0]))
        rootAng = quat.mul_vec(rootRot, quat.inv_mul_vec(Yrot[animFrame,0], Yang[animFrame,0]))
        rootPos = rootPos + rootVel * deltaTime
        rootRot = quat.mul(quat.from_scaled_angle_axis(rootAng * deltaTime), rootRot)

        # Update the rest of the pose using inertialization
        
        inertHalflife = 0.075
        
        offPos, offVel, outPos, _ = InertializeUpdatePosition(
            offPos, offVel,
            Ypos[animFrame,1:], Yvel[animFrame,1:], inertHalflife, deltaTime)
            
        offRot, offAng, outRot, _ = InertializeUpdateRotation(
            offRot, offAng,
            Yrot[animFrame,1:], Yang[animFrame,1:], inertHalflife, deltaTime)
        
        locRot = np.concatenate([rootRot[None], outRot], axis=0)
        locPos = np.concatenate([rootPos[None], outPos], axis=0)
        
        # Computing fk using transformation matrices is much more efficient since it minimizes python overhead
        
        locXforms = np.zeros([len(locRot), 4, 4])
        locXforms[...,:3,:3] = quat.to_xform(locRot)
        locXforms[...,:3,3] = locPos
        locXforms[...,3,3] = 1
        
        gloXforms = locXforms.copy()
        for i in range(1, len(parents)):
            gloXforms[...,i,:,:] = gloXforms[...,parents[i],:,:] @ locXforms[...,i,:,:]
        
        gloRot = quat.from_xform(gloXforms)
        gloPos = gloXforms[...,:3,3]
        
        # Update pose on mesh
        
        UpdateModelPoseFromNumpyArrays(genoModel, bindPos, bindRot, gloPos[1:], gloRot[1:])

        # Shadow Light Tracks Character
        
        hipPosition = Vector3(*gloPos[1])
        
        shadowLight.target = Vector3(hipPosition.x, 0.0, hipPosition.z)
        shadowLight.position = Vector3Add(shadowLight.target, Vector3Scale(lightDir, -5.0))

        # Update Camera
        
        camera.update(
            Vector3(hipPosition.x, 0.75, hipPosition.z),
            GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
            GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
            GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
            GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
            GetMouseWheelMove(),
            deltaTime)
        
        # Render
        
        rlDisableColorBlend()
        
        BeginDrawing()
        
        # Render Shadow Maps
        
        BeginShadowMap(shadowMap, shadowLight)  
        
        lightViewProj = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection())
        lightClipNearPtr[0] = rlGetCullDistanceNear()
        lightClipFarPtr[0] = rlGetCullDistanceFar()
        
        SetShaderValue(shadowShader, shadowShaderLightClipNear, lightClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(shadowShader, shadowShaderLightClipFar, lightClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedShadowShader, skinnedShadowShaderLightClipNear, lightClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedShadowShader, skinnedShadowShaderLightClipFar, lightClipFarPtr, SHADER_UNIFORM_FLOAT)
        
        groundModel.materials[0].shader = shadowShader
        DrawModel(groundModel, groundPosition, 1.0, WHITE)
        
        genoModel.materials[0].shader = skinnedShadowShader
        DrawModel(genoModel, genoPosition, 1.0, WHITE)
        
        EndShadowMap()
        
        # Render GBuffer
        
        BeginGBuffer(gbuffer, camera.cam3d)
        
        camView = rlGetMatrixModelview()
        camProj = rlGetMatrixProjection()
        camInvProj = MatrixInvert(camProj)
        camInvViewProj = MatrixInvert(MatrixMultiply(camView, camProj))
        camClipNearPtr[0] = rlGetCullDistanceNear()
        camClipFarPtr[0] = rlGetCullDistanceFar()
        specularityPtr[0] = 0.5
        glossinessPtr[0] = 10.0
        
        SetShaderValue(basicShader, basicShaderSpecularity, specularityPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(basicShader, basicShaderGlossiness, glossinessPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(basicShader, basicShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(basicShader, basicShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderSpecularity, specularityPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderGlossiness, glossinessPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(skinnedBasicShader, skinnedBasicShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)        
        
        groundModel.materials[0].shader = basicShader
        DrawModel(groundModel, groundPosition, 1.0, Color(190, 190, 190, 255))
        
        genoModel.materials[0].shader = skinnedBasicShader
        DrawModel(genoModel, genoPosition, 1.0, ORANGE)       
        
        EndGBuffer(screenWidth, screenHeight)
        
        # Render SSAO and Shadows
        
        BeginTextureMode(ssaoFront)
        
        BeginShaderMode(ssaoShader)
        
        SetShaderValueTexture(ssaoShader, ssaoShaderGBufferNormal, gbuffer.normal)
        SetShaderValueTexture(ssaoShader, ssaoShaderGBufferDepth, gbuffer.depth)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamView, camView)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamProj, camProj)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamInvProj, camInvProj)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamInvViewProj, camInvViewProj)
        SetShaderValueMatrix(ssaoShader, ssaoShaderLightViewProj, lightViewProj)
        SetShaderValueShadowMap(ssaoShader, ssaoShaderShadowMap, shadowMap)
        SetShaderValue(ssaoShader, ssaoShaderShadowInvResolution, ffi.addressof(shadowInvResolution), SHADER_UNIFORM_VEC2)
        SetShaderValue(ssaoShader, ssaoShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderLightClipNear, lightClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderLightClipFar, lightClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderLightDir, ffi.addressof(lightDir), SHADER_UNIFORM_VEC3)
        
        ClearBackground(WHITE)
        
        DrawTextureRec(
            ssaoFront.texture,
            Rectangle(0, 0, ssaoFront.texture.width, -ssaoFront.texture.height),
            Vector2(0.0, 0.0),
            WHITE)

        EndShaderMode()

        EndTextureMode()
        
        # Blur Horizontal
        
        BeginTextureMode(ssaoBack)
        
        BeginShaderMode(blurShader)
        
        blurDirection = Vector2(1.0, 0.0)
        blurInvTextureResolution = Vector2(1.0 / ssaoFront.texture.width, 1.0 / ssaoFront.texture.height)
        
        SetShaderValueTexture(blurShader, blurShaderGBufferNormal, gbuffer.normal)
        SetShaderValueTexture(blurShader, blurShaderGBufferDepth, gbuffer.depth)
        SetShaderValueTexture(blurShader, blurShaderInputTexture, ssaoFront.texture)
        SetShaderValueMatrix(blurShader, blurShaderCamInvProj, camInvProj)
        SetShaderValue(blurShader, blurShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(blurShader, blurShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(blurShader, blurShaderInvTextureResolution, ffi.addressof(blurInvTextureResolution), SHADER_UNIFORM_VEC2)
        SetShaderValue(blurShader, blurShaderBlurDirection, ffi.addressof(blurDirection), SHADER_UNIFORM_VEC2)

        DrawTextureRec(
            ssaoBack.texture,
            Rectangle(0, 0, ssaoBack.texture.width, -ssaoBack.texture.height),
            Vector2(0, 0),
            WHITE)

        EndShaderMode()

        EndTextureMode()
      
        # Blur Vertical
        
        BeginTextureMode(ssaoFront)
        
        BeginShaderMode(blurShader)
        
        blurDirection = Vector2(0.0, 1.0)
        
        SetShaderValueTexture(blurShader, blurShaderInputTexture, ssaoBack.texture)
        SetShaderValue(blurShader, blurShaderBlurDirection, ffi.addressof(blurDirection), SHADER_UNIFORM_VEC2)

        DrawTextureRec(
            ssaoFront.texture,
            Rectangle(0, 0, ssaoFront.texture.width, -ssaoFront.texture.height),
            Vector2(0, 0),
            WHITE)

        EndShaderMode()

        EndTextureMode()
      
        # Light GBuffer
        
        BeginTextureMode(lighted)
        
        BeginShaderMode(lightingShader)
        
        sunColor = Vector3(253.0 / 255.0, 255.0 / 255.0, 232.0 / 255.0)
        sunStrengthPtr[0] = 0.25
        skyColor = Vector3(174.0 / 255.0, 183.0 / 255.0, 190.0 / 255.0)
        skyStrengthPtr[0] = 0.15
        groundStrengthPtr[0] = 0.1
        ambientStrengthPtr[0] = 1.0
        exposurePtr[0] = 0.9
        
        SetShaderValueTexture(lightingShader, lightingShaderGBufferColor, gbuffer.color)
        SetShaderValueTexture(lightingShader, lightingShaderGBufferNormal, gbuffer.normal)
        SetShaderValueTexture(lightingShader, lightingShaderGBufferDepth, gbuffer.depth)
        SetShaderValueTexture(lightingShader, lightingShaderSSAO, ssaoFront.texture)
        SetShaderValue(lightingShader, lightingShaderCamPos, ffi.addressof(camera.cam3d.position), SHADER_UNIFORM_VEC3)
        SetShaderValueMatrix(lightingShader, lightingShaderCamInvViewProj, camInvViewProj)
        SetShaderValue(lightingShader, lightingShaderLightDir, ffi.addressof(lightDir), SHADER_UNIFORM_VEC3)
        SetShaderValue(lightingShader, lightingShaderSunColor, ffi.addressof(sunColor), SHADER_UNIFORM_VEC3)
        SetShaderValue(lightingShader, lightingShaderSunStrength, sunStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderSkyColor, ffi.addressof(skyColor), SHADER_UNIFORM_VEC3)
        SetShaderValue(lightingShader, lightingShaderSkyStrength, skyStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderGroundStrength, groundStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderAmbientStrength, ambientStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderExposure, exposurePtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        
        ClearBackground(RAYWHITE)
        
        DrawTextureRec(
            gbuffer.color,
            Rectangle(0, 0, gbuffer.color.width, -gbuffer.color.height),
            Vector2(0, 0),
            WHITE)
        
        EndShaderMode()        
        
        # Debug Draw
        
        BeginMode3D(camera.cam3d)
        
        # DrawTransform(rootPos, rootRot, 0.1)
        
        #DrawCapsule(Vector3(*rootPos), Vector3(*(rootPos + desiredDir)), 0.01, 5, 7, BLUE)
        #DrawCapsule(Vector3(*rootPos), Vector3(*(rootPos + desiredVel)), 0.01, 5, 7, GREEN)
        DrawTrajectory(Tpos, Tdir, RED);
        
        # DrawFeatures(X[animFrame], Xoffset, Xscale, rootPos, rootRot, PINK)
        # DrawFeatures(Xquery, Xoffset, Xscale, rootPos, rootRot, ORANGE)
        
        EndMode3D()

        EndTextureMode()
        
        # Render Final with FXAA
        
        BeginShaderMode(fxaaShader)

        fxaaInvTextureResolution = Vector2(1.0 / lighted.texture.width, 1.0 / lighted.texture.height)
        
        SetShaderValueTexture(fxaaShader, fxaaShaderInputTexture, lighted.texture)
        SetShaderValue(fxaaShader, fxaaShaderInvTextureResolution, ffi.addressof(fxaaInvTextureResolution), SHADER_UNIFORM_VEC2)
        
        DrawTextureRec(
            lighted.texture,
            Rectangle(0, 0, lighted.texture.width, -lighted.texture.height),
            Vector2(0, 0),
            WHITE)
        
        EndShaderMode()
  
        # UI
  
        rlEnableColorBlend()
  
        GuiGroupBox(Rectangle(20, 10, 190, 180), b"Camera")

        GuiLabel(Rectangle(30, 20, 150, 20), b"Ctrl + Left Click - Rotate")
        GuiLabel(Rectangle(30, 40, 150, 20), b"Ctrl + Right Click - Pan")
        GuiLabel(Rectangle(30, 60, 150, 20), b"Mouse Scroll - Zoom")
        GuiLabel(Rectangle(30, 80, 150, 20), b"Target: [% 5.3f % 5.3f % 5.3f]" % (camera.cam3d.target.x, camera.cam3d.target.y, camera.cam3d.target.z))
        GuiLabel(Rectangle(30, 100, 150, 20), b"Offset: [% 5.3f % 5.3f % 5.3f]" % (camera.offset.x, camera.offset.y, camera.offset.z))
        GuiLabel(Rectangle(30, 120, 150, 20), b"Azimuth: %5.3f" % camera.azimuth)
        GuiLabel(Rectangle(30, 140, 150, 20), b"Altitude: %5.3f" % camera.altitude)
        GuiLabel(Rectangle(30, 160, 150, 20), b"Distance: %5.3f" % camera.distance)
  
        EndDrawing()

    UnloadRenderTexture(lighted)
    UnloadRenderTexture(ssaoBack)
    UnloadRenderTexture(ssaoFront)
    UnloadRenderTexture(lighted)
    UnloadGBuffer(gbuffer)

    UnloadShadowMap(shadowMap)
    
    UnloadModel(genoModel)
    UnloadModel(groundModel)
    
    UnloadShader(fxaaShader)    
    UnloadShader(blurShader)    
    UnloadShader(ssaoShader) 
    UnloadShader(lightingShader)    
    UnloadShader(basicShader)
    UnloadShader(skinnedBasicShader)
    UnloadShader(skinnedShadowShader)
    UnloadShader(shadowShader)
    
    CloseWindow()