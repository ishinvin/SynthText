import numpy as np

class DepthCamera(object):
    """
    Camera functions for Depth-CNN camera.
    """
    f = 520

    @staticmethod
    def plane2xyz(center, ij, plane):
        """
        converts image pixel indices to xyz on the PLANE.

        center : 2-tuple
        ij : nx2 int array
        plane : 4-tuple

        return nx3 array.
        """
        ij = np.atleast_2d(ij)
        n = ij.shape[0]
        ij = ij.astype('float')
        xy_ray = (ij-center[None,:]) / DepthCamera.f
        z = -plane[2]/(xy_ray.dot(plane[:2])+plane[3])
        xyz = np.c_[xy_ray, np.ones(n)] * z[:,None]
        return xyz

    @staticmethod
    def depth2xyz(depth):
        """
        Convert a HxW depth image (float, in meters)
        to XYZ (HxWx3).

        y is along the height.
        x is along the width.
        """
        H,W = depth.shape
        xx,yy = np.meshgrid(np.arange(W),np.arange(H))
        X = (xx-W/2) * depth / DepthCamera.f
        Y = (yy-H/2) * depth / DepthCamera.f
        return np.dstack([X,Y,depth.copy()])