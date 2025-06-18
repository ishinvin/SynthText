import numpy as np

def fit_plane(xyz,z_pos=None):
    """
    if z_pos is not None, the sign
    of the normal is flipped to make 
    the dot product with z_pos (+).
    """
    mean = np.mean(xyz,axis=0)
    xyz_c = xyz - mean[None,:]
    l,v = np.linalg.eig(xyz_c.T.dot(xyz_c))
    abc = v[:,np.argmin(l)]
    d = -np.sum(abc*mean)
    # unit-norm the plane-normal:
    abcd =  np.r_[abc,d]/np.linalg.norm(abc)
    # flip the normal direction:
    if z_pos is not None:
        if np.sum(abcd[:3]*z_pos) < 0.0:
            abcd *= -1
    return abcd

def fit_plane_ransac(pts, neighbors=None,z_pos=None, dist_inlier=0.05, 
                     min_inlier_frac=0.60, nsample=3, max_iter=100):
    """
    Fits a 3D plane model using RANSAC. 
    pts : (nx3 array) of point coordinates   
    """
    n,_ = pts.shape
    ninlier,models = [],[]
    for i in range(max_iter):
        if neighbors is None:
            p = pts[np.random.choice(pts.shape[0],nsample,replace=False),:]
        else:
            p = pts[neighbors[:,i],:]
        m = fit_plane(p,z_pos)
        ds = np.abs(pts.dot(m[:3])+m[3])
        nin = np.sum(ds < dist_inlier)
        if nin/pts.shape[0] >= min_inlier_frac:
            ninlier.append(nin)
            models.append(m)

    if models == []:
        print ("RANSAC plane fitting failed!")
        return #None
    else: #refit the model to inliers:
        ninlier = np.array(ninlier)
        best_model_idx = np.argsort(-ninlier)
        n_refit, m_refit, inliers = [],[],[]
        for idx in best_model_idx[:min(10,len(best_model_idx))]:
            # re-estimate the model based on inliers:
            dists = np.abs(pts.dot(models[idx][:3])+models[idx][3])
            inlier = dists < dist_inlier
            m = fit_plane(pts[inlier,:],z_pos)
            # compute new inliers:
            d = np.abs(pts.dot(m[:3])+m[3])
            inlier = d < dist_inlier/2 # heuristic
            n_refit.append(np.sum(inlier))
            m_refit.append(m)
            inliers.append(inlier)
        best_plane = np.argmax(n_refit)
        return m_refit[best_plane],inliers[best_plane]