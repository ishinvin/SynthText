import numpy as np 
from ransac import fit_plane_ransac

def ensure_proj_z(plane_coeffs, min_z_proj):
    a,b,c,d = plane_coeffs
    if np.abs(c) < min_z_proj:
        s = ((1 - min_z_proj**2) / (a**2 + b**2))**0.5
        coeffs = np.array([s*a, s*b, np.sign(c)*min_z_proj, d])
        assert np.abs(np.linalg.norm(coeffs[:3])-1) < 1e-3
        return coeffs
    return plane_coeffs

def isplanar(xyz,sample_neighbors,dist_thresh,num_inliers,z_proj):
    """
    Checks if at-least FRAC_INLIERS fraction of points of XYZ (nx3)
    points lie on a plane. The plane is fit using RANSAC.

    XYZ : (nx3) array of 3D point coordinates
    SAMPLE_NEIGHBORS : 5xN_RANSAC_TRIALS neighbourhood array
                       of indices into the XYZ array. i.e. the values in this
                       matrix range from 0 to number of points in XYZ
    DIST_THRESH (default = 10cm): a point pt is an inlier iff dist(plane-pt)<dist_thresh
    FRAC_INLIERS : fraction of total-points which should be inliers to
                   to declare that points are planar.
    Z_PROJ : changes the surface normal, so that its projection on z axis is ATLEAST z_proj.

    Returns:
        None, if the data is not planar, else a 4-tuple of plane coeffs.
    """
    frac_inliers = num_inliers/xyz.shape[0]
    dv = -np.percentile(xyz,50,axis=0) # align the normal to face towards camera
    max_iter = sample_neighbors.shape[-1]
    plane_info =  fit_plane_ransac(xyz,neighbors=sample_neighbors,
                            z_pos=dv,dist_inlier=dist_thresh,
                            min_inlier_frac=frac_inliers,nsample=20,
                            max_iter=max_iter) 
    if plane_info != None:
        coeff, inliers = plane_info
        coeff = ensure_proj_z(coeff, z_proj)
        return coeff,inliers
    else:
        return #None


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

def ssc(v):
    """
    Returns the skew-symmetric cross-product matrix corresponding to v.
    """
    v /= np.linalg.norm(v)
    return np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]])

def rot3d(v1,v2):
    """
    Rodrigues formula : find R_3x3 rotation matrix such that v2 = R*v1.
    https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula#Matrix_notation
    """
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    v3 = np.cross(v1,v2)
    s = np.linalg.norm(v3)
    c = v1.dot(v2)
    Vx = ssc(v3)
    return np.eye(3)+s*Vx+(1-c)*Vx.dot(Vx)

def unrotate2d(pts):
    """
    PTS : nx3 array
    finds principal axes of pts and gives a rotation matrix (2d)
    to realign the axes of max variance to x,y.
    """
    mu = np.median(pts,axis=0)
    pts -= mu[None,:]
    l,R = np.linalg.eig(pts.T.dot(pts))
    R = R / np.linalg.norm(R,axis=0)[None,:]

    # make R compatible with x-y axes:
    if abs(R[0,0]) < abs(R[0,1]): #compare dot-products with [1,0].T
        R = np.fliplr(R)
    if not np.allclose(np.linalg.det(R),1):
        if R[0,0]<0:
            R[:,0] *= -1
        elif R[1,1]<0:
            R[:,1] *= -1
        else:
            print ("Rotation matrix not understood")
            return
    if R[0,0]<0 and R[1,1]<0:
        R *= -1
    assert np.allclose(np.linalg.det(R),1)

    # at this point "R" is a basis for the original (rotated) points.
    # we need to return the inverse to "unrotate" the points:
    return R.T #return the inverse
