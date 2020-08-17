import numpy as np

from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights
)

# File: Pytorch3DObjectGenerator
# Classes: 
#          ConvertMesh
# Functions:
#          createRenderer
#          saveImage
#          openImage

class ConvertMesh():
    
    def __init__(self,mesh,renderer,initial_camera_position=np.array([3.0,50.0,0.0])):
        
        # Function: __init__
        # Inputs:   self,mesh,renderer,initial_camera_position=np.array([3.0,50.0,0.0])
        # Process:  creates class
        
        super().__init__()
        self.mesh                    = mesh
        self.renderer                = renderer 
        
        #Units are distance, elevation, azimuth (angle)
        self.camera_position         = initial_camera_position
        
    def set_camera_position(self,camera_position):
        
        # Function: set_camera_position
        # Inputs:   self,camera_position
        # Process:  sets camera_position
        
        self.camera_position = camera_position
        
    def get_camera_position(self):
        
        # Function: get_camera_position
        # Inputs:   self
        # Process:  returns camera position
        
        return self.camera_position
    
    def renderImage(self):
        
        # Function: renderImage
        # Inputs:   self
        # Process:  returns image of the object
        
        R, T = look_at_view_transform(self.camera_position[0], self.camera_position[1], self.camera_position[2])
        image = self.renderer(meshes_world=self.mesh.clone(),R=R,T=T)
        
        return image
    
def createRenderer(image_size,faces_per_pixel,lights_location):
    
    # Function: createRenderer
    # Inputs:   image_size,faces_per_pixel,lights_location
    # Process:  creates an image renderer
    # Output:   returns renderer
        
    cameras = OpenGLPerspectiveCameras()
    
    #Settings for Raster
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=faces_per_pixel, 
    )

    # We can add a point light in front of the object. 
    lights = PointLights(location=(lights_location,))
    created_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(cameras=cameras, lights=lights)
    )
    
    return created_renderer

def saveImage(image,path):
    
    # Function: saveImage
    # Inputs:   image,path
    # Process:  saves image to path 
    
    input_image  = np.uint8(image[:,:,:,0:3].numpy().squeeze()*255)
    image_output = Image.fromarray(input_image)
    
    image_output.save(path)
    
def openImage(path):
    
    # Function: openImage
    # Inputs:   path
    # Process:  opens image in path
    # Output:   returns image
    
    image = Image.open(path)
    image.load()
    
    imageData = torch.Tensor(np.array(image, dtype='f8')).transpose(0,2).unsqueeze(0)
    
    return imageData



