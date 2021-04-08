import numpy as np
import matplotlib as mpl
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import cv2
import pyvista as pv
import open3d as o3d
import time
import etch_sim_utilities_v3 as utils
import scipy.signal as sg
import os
from matplotlib.path import Path
from mpl_toolkits.mplot3d import axes3d
from shapely.geometry.polygon import Polygon
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
from sys import getsizeof
from scipy import ndimage
from itertools import product


mpl.style.use('default')

"""
Written by Nicolas Castano with contributions from Seth Cordts

Model continuous etching into silicon wafer on the PT-DSE tool in the SNF
based known etch rates.

Data is stored in ordered dictionary with keys as specific step, kept in the
order in which it was created:
    etch_grid = {'init': [pv.PolyData(mask_cont_0), pv.PolyData(mask_cont_1),
                          pv.PolyData(mask_cont_2), ...],
                 global_step_0: [pv.PolyData(mask_cont_0), ne
                                 pv.PolyData(mask_cont_1),
                                 pv.PolyData(mask_cont_2), ...],
                 global_step_N: [pv.PolyData(mask_cont_0)]}

"""


# %% define recipe
# ex: {'step1':{'bosch':13,'iso':100,'cylces':7},
#      'step2':{'bosch':240,'iso':None,'cycles':240},
#      'step3':{'bosch':None,'iso':70,'cycles':1}}
#recipe_steps = {'step1':{'bosch':13,'iso':100,'cycles':7},
#                'step2':{'bosch':240,'iso':None,'cycles':240},
#                'step3':{'bosch':None,'iso':70,'cycles':1}}
#recipe_steps = {'step0':{'bosch':7,'iso':5,'cycles':2}}
#recipe_steps = {'step01':{'bosch':15,'iso':100,'cycles':7},
##                'step02':{'bosch':300,'iso':None,'cycles':300},
#                'step03':{'bosch':None,'iso':100,'cycles':1}}
recipe_steps = {'step01':{'bosch':12,'iso':100,'cycles':4},
                'step02':{'bosch':None,'iso':1000,'cycles':1}}
# recipe_steps = {'step01':{'bosch':None,'iso':700,'cycles':1}}
# recipe_steps = {'step00':{'bosch':40,'iso':None,'cycles':40},
#                 'step01':{'bosch':None,'iso':525,'cycles':1}}


# %% user inputs
# load mask
master_dir = 'C:/Users/Seth Cordts/OneDrive - Leland Stanford Junior University/TangLab/Papers/microDicer/Simulation/'

im_dir = master_dir + 'masks/'
im_file = 'mask_WXC10R7_rot45.png'


horiz_to_vert_rate_ratio_sweep = [0.8, 0.6]
alpha_f_sweep = [0.5]
cell_size_sweep = [7]  # microns
C_etch_sweep    = [1.9]
params = [horiz_to_vert_rate_ratio_sweep, alpha_f_sweep, cell_size_sweep, C_etch_sweep]
all_param_combos = list(product(*params))
KOI = ['step01_bosch-iso01_bosch002_isotime0', 'step01_bosch-iso04_bosch012_isotime104', 'step02_iso01_isotime320', 'step02_iso01_isotime800', 'step02_iso01_isotime992', 'step02_iso01_isotime1000']

print(all_param_combos)
for par in all_param_combos:
    horiz_to_vert_rate_ratio = par[0]
    alpha_f = par[1]
    cell_size = par[2]
    C_etch = par[3]
    
        
    plot_state_dir = master_dir + 'v12_2serr_states_'+ im_file.split('.png')[0]+str(horiz_to_vert_rate_ratio)+"_"+str(alpha_f)+"_"+str(cell_size)+"_"+str(C_etch)+"/"
    if not os.path.exists(plot_state_dir):
        os.makedirs(plot_state_dir)
    plot_z_dir = master_dir + 'v12_2serr_z_'+im_file.split('.png')[0]+str(horiz_to_vert_rate_ratio)+"_"+str(alpha_f)+"_"+str(cell_size)+"_"+str(C_etch)+"/"
    if not os.path.exists(plot_z_dir):
        os.makedirs(plot_z_dir)
    
    pixel_um_conv = 212/100  # mask_WXC7R5
   # pixel_um_conv = 236/148
    # pixel_um_conv = 97.5/73.5  # mask_WXC7R5_rot45
    theta_rot = np.pi/4
    
    # read in mask image and define contour
    im_path = im_dir + im_file
    curr_im = cv2.imread(im_path, cv2.IMREAD_ANYDEPTH)
    curr_im = cv2.GaussianBlur(curr_im,(3,3),0)
    rgb_im = cv2.cvtColor(curr_im, cv2.COLOR_GRAY2RGB)
    conts, hier = cv2.findContours(curr_im, cv2.RETR_LIST,
                                            cv2.CHAIN_APPROX_NONE)
    conts_im = cv2.drawContours(rgb_im, conts, -1, (0,255,0),3)
    # show the contour to verify
    dummy_i = im_file.find('.png')
    out_file = im_dir + im_file[:dummy_i] + '_out' + im_file[dummy_i:]
    cv2.imwrite(out_file, conts_im)
    
    cell_vol = 1#cell_size**3
    wafer_thickness = 525  # microns
    
    center_to_surface = wafer_thickness - cell_size/2
    
    h = curr_im.shape[0]
    w = curr_im.shape[1]
    
    contour_read_step = 5
    topo_im = np.zeros_like(curr_im)
    window_len = 7  # for smoothing of mask contour read
    
    # smoothing kernel
    kernel_shape = (3,3)
    kernel = np.full(kernel_shape, 1/np.prod(kernel_shape))
    kern = np.array([[1,1,1],[1,1,1],[1,1,1]])
    
    
    
    # %% etch rate functions
    
    a = 0.1715
    b = 0.001
    def vert_rate(z):
        # a = 0.5
        a = 0.1715
        b = 0.001
        return a*np.exp(-b*z)
    
    def horiz_rate(z):
        return horiz_to_vert_rate_ratio*vert_rate(z)
    
    def bosch_vert_step(z):
        return 0.84 - 0.1/wafer_thickness*z
    
    #bosch_vert_step = 0.84  # um/step
    
    #horiz_rate = 0.09# vert_rate*0.6#90/600 # vert_rate*0.6  # um/s
    
    # %% determine time step from cell size and etch rate
    t_start = 0 # seconds
    t_step = np.int(np.floor(0.3*cell_size/vert_rate(0)))
    
    
    param_dict = {"recipe_steps":recipe_steps, "im_file" :im_file, "pixel_um_conv =":pixel_um_conv, "theta_rot":theta_rot, "alpha_f":alpha_f,  "cell_size":cell_size, "horiz_to_vert_rate_ratio":horiz_to_vert_rate_ratio, "etch rate (a)":a, "etch rate(b)":b,"C_etch":C_etch,"t_step":t_step}
    paramName = os.path.join(plot_z_dir, "parameters.txt")
    with open(paramName, 'w')as f:
        for key, value in param_dict.items():
            print(key, ' : ', value, file = f)
    
    # %% make mask paths
    
    # initialize global topo data container following data structure
    # indicated in the script header
    
    # construct global data container; this is a ordered dictionary so later we
    # can loop over keys and ensure that
    etch_grid, total_iso_time = utils.define_steps(recipe_steps, t_start, t_step)
    n_steps = len(list(etch_grid.keys()))
    
    neigh_mem = []
    exp_mem = []
    rm_mem = []
    
    
    # construct mask paths and check cell centers are within masks
    # path objects used for determining if point is within mask
    mask_paths = {}
    # build initial geometries from mask that will be tracked through solution
    print('building initial features')
    x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
    for c, cont in enumerate(conts):
        x = []
        y = []
        # gather points in mask contours
        for p, point in enumerate(cont):
            if p%contour_read_step== 0:
                # translate point so mask centered at 0,0
                temp_x = point[0][0]/pixel_um_conv - (w/pixel_um_conv)/2
                temp_y = point[0][1]/pixel_um_conv - (h/pixel_um_conv)/2
                x.append(temp_x)
                y.append(temp_y)
                if temp_x < x_min: x_min = temp_x
                if temp_x > x_max: x_max = temp_x
                if temp_y > y_max: y_max = temp_y
                if temp_y < y_min: y_min = temp_y
    
    
        # force last point to be on top of first point to close the polygon
        # remove redundant points
        x[-1] = x[0]
        y[-1] = y[0]
        # smooth contour with spline
        points = np.hstack((np.reshape(np.array(x),[len(x),1]),
                           np.reshape(np.array(y),[len(y),1])))
        tck, u = splprep(points.T, u=None, s=0.0, per=1)
        u_new = np.linspace(u.min(), u.max(), len(cont))
        x_spline, y_spline = splev(u_new, tck, der=0)
        points = np.hstack((np.reshape(np.array(x_spline),[len(x_spline),1]),
                            np.reshape(np.array(y_spline),[len(y_spline),1])))
        # make polygon objects
        mask_poly = Polygon(points)
        mask_poly.exterior
        # make path objects (has nice contains_points method)
        mask_paths[c] = Path(mask_poly.exterior,closed=True)
        # this just means theres no buffer region around the feature
        buff_mask = mask_paths[c]
    
    
    # %% initilalize arrays for states, coords, and norms
    
    # if abs(x_min) > x_max:
    #     x_max = abs(x_min)
    # else:
    #     x_min = -x_max
    # if abs(y_min) > y_max:
    #     y_max = abs(y_min)
    # else:
    #     y_min = -y_max
    
    x_mid = x_min + (x_max - x_min)/2
    y_mid = y_min + (y_max - y_min)/2
    
    
    # adjust cell_size
    n_indices = np.ceil((x_max - x_min)/cell_size)
    cell_size = (x_max - x_min)/n_indices
    
    buffer = horiz_rate(0) * (total_iso_time*7)
    n_indices = np.ceil(buffer/cell_size)
    buffer = n_indices*cell_size
    
    x_axis = np.arange(x_min-cell_size-buffer,
                        x_max+cell_size+buffer,
                        cell_size,dtype=np.float32)
    y_axis = np.arange(y_min-cell_size-buffer,
                        y_max+cell_size+buffer,
                        cell_size,dtype=np.float32)
    
    # x_axis = np.arange(x_max+cell_size+buffer,
    #                    x_min-cell_size-buffer,
    #                    -cell_size,dtype=np.float32)
    # y_axis = np.arange(y_max+cell_size+buffer,
    #                    y_min-cell_size-buffer,
    #                    -cell_size,dtype=np.float32)
    
    
    z_axis = np.arange(0-cell_size,
                       wafer_thickness+cell_size,
                       cell_size,dtype=np.float32)
    
    # first concatenate paths for easy min dist operations
    concat_mask_verts = None
    for c in mask_paths:
        pts = np.hstack((mask_paths[c].vertices,
                        np.ones((mask_paths[c].vertices.shape[0],1)) * z_axis[-2]))
        try:
            concat_mask_verts = np.vstack((concat_mask_verts,pts))
        except:
            concat_mask_verts = pts
    
    coords = np.meshgrid(x_axis, y_axis, z_axis)
    
    # rotate 45 degrees for rot45 mask
    temp_x = np.cos(theta_rot)*coords[0] - np.sin(theta_rot)*coords[1]
    temp_y = np.sin(theta_rot)*coords[0] + np.cos(theta_rot)*coords[1]
    coords[0] = temp_x
    coords[1] = temp_y
    
    grid_shape = coords[0].shape
    
    states = np.ones_like(coords[0],dtype=np.int) * 2
    phis = np.ones_like(coords[0],dtype=np.float32) * 10
    
    norms = np.empty(grid_shape,dtype=object)
    ii = range(grid_shape[0])
    jj = range(grid_shape[1])
    kk = range(grid_shape[2])
    
     # set the top slice to -1
    exposed_indices = []
    exposed_indices_set = set(exposed_indices)
    # cordinates in the x, y confines of mask and weights for bosch etching
    known_in_mask_indices = {}
    rm_indices = []
    neigh_indices = set()
    
    #### add cells to exposed list
    # check for cells whose centers are clearly in the mask
    for i in ii:
        if i % int(grid_shape[0]/10) == 0:
                print('checking in mask points, row %i of %i' %
                      (i,grid_shape[0]))
        for j in jj:
            pt = [coords[0][i,j,0], coords[1][i,j,0]]
            in_mask = utils.is_in_mask(pt[0],pt[1],mask_paths)
            # if the cell center is clearly in mask path save and assign phi
            if in_mask == True:
                temp_tuple = tuple(pt[0:2])
                known_in_mask_indices[tuple([i,j])] = 1
                # initial normals pointing up
                norms[i,j,-2] = np.array([0,0,1])
                # add to exposed_coords set
                exposed_indices.append([i,j,kk[-2]])
                exposed_indices_set.add(tuple([i,j,kk[-2]]))
    
                # initialize state by signed distance from the mask etch front
                states[i,j,kk[-2]] = 0
                states[i,j,kk[-1]] = -1
                states[i,j,kk[-3]] = 1
    
                phis[i,j,kk[-2]] = 0.5*cell_size
    
    # adjust phi for initial exposed cells (ie the boundary phis)
    dummy = phis
    mp = concat_mask_verts
    for ind_i, index in enumerate(exposed_indices):
        i,j,k = index[0], index[1], index[2]
        x,y,z = coords[0][i,j,k], coords[1][i,j,k], coords[2][i,j,k]
        dist = np.sqrt(np.sum((mp - np.array([x,y,z]))**2, axis=1))
        min_dist_i = dist.argmin()
        if dist[min_dist_i] < abs(phis[i,j,k]):
            dummy[i,j,k] = dist[min_dist_i]
            known_in_mask_indices[tuple([i,j])] = 1/(cell_size/2) * dummy[i,j,k]
    phis = dummy
    
    
    #### add cells to neighbors
    dummy = phis
    for index in exposed_indices:
        i,j,k = index[0], index[1], index[2]
        potential_neigh = utils.list_neighbor_indices([i,j,k])
        for neigh in potential_neigh:
            if tuple(neigh) not in exposed_indices_set and \
                neigh[2] != kk[-1]:
    
                i_n,j_n,k_n = neigh[0], neigh[1], neigh[2]
                neigh_indices.add(tuple(neigh))
                states[i_n,j_n,k_n] = 1
    
                # determine phi value to assign
                nidx = utils.list_neighbor_indices(neigh)
                min_phi = np.inf
                for idx in nidx:
                    i_en,j_en,k_en = idx[0], idx[1], idx[2]
                    if tuple(idx) in exposed_indices_set:
                        if phis[i_en,j_en,k_en] < min_phi:
                            min_phi = phis[i_en,j_en,k_en]
                dummy[i_n,j_n,k_n] = min_phi + cell_size
    phis = dummy
    
    neigh_indices = list(neigh_indices)
    
    # determine neighbor signed distance
    dummy = phis
    for ind_i,index in enumerate(neigh_indices):
        if ind_i % int(len(neigh_indices)/10) == 0:
            print('assigning neighbor signed distances, %i of %i' %
                  (ind_i+1,len(neigh_indices)))
        i,j,k = index[0], index[1], index[2]
        # input mask paths if the neighbor cell is on the surface
        if k == kk[-2]:
            mp = concat_mask_verts
        else:
            mp = None
        distance = utils.min_dist_to_etch_front(index, exposed_indices, coords,
                                                mp=mp)
        dummy[i,j,k] = distance
    phis = dummy
    
    # #### check all neighbor phi to see if any should be converted to exposed
    # neigh_indices_set = set((i[0],i[1],i[2]) for i in neigh_indices)
    
    # for index in neigh_indices:
    #     i,j,k = index[0], index[1], index[2]
    #     if phis[i,j,k] < cell_size/2:
    #         states[i,j,k] = 0
    #         exposed_indices.append(index)
    #         neigh_indices_set.remove(index)
    #         known_in_mask_indices[tuple([i,j])] = 1/(cell_size/2) * phis[i,j,k]
    
    # #### and refresh neighbors again
    # dummy = phis
    # for index in exposed_indices:
    #     i,j,k = index[0], index[1], index[2]
    #     potential_neigh = utils.list_neighbor_indices([i,j,k])
    #     for neigh in potential_neigh:
    #         if tuple(neigh) not in exposed_indices_set and \
    #             neigh[2] != kk[-1]:
    
    #             i_n,j_n,k_n = neigh[0], neigh[1], neigh[2]
    #             neigh_indices_set.add(tuple(neigh))
    #             states[i_n,j_n,k_n] = 1
    #             # determine phi value to assign
    #             nidx = utils.list_neighbor_indices(neigh)
    #             min_phi = cell_size + cell_size/2
    #             flag = False
    #             for idx in nidx:
    #                 i_en,j_en,k_en = idx[0], idx[1], idx[2]
    #                 if tuple(idx) in exposed_indices_set:
    #                     if phis[i_en,j_en,k_en] < min_phi:
    #                         min_phi = phis[i_en,j_en,k_en]
    #                         flag = True
    
    #             if flag is True:
    #                 dummy[i_n,j_n,k_n] = min_phi + cell_size
    #             else:
    #                 dummy[i_n,j_n,k_n] = min_phi
    
    # phis = dummy
    # neigh_indices = list(neigh_indices_set)
    
    
    
    # %% loop over steps
    step_index_lookup = {i:key for i,key in enumerate(etch_grid)}
    print('')
    # loop over etch_grid keys (after init) which represent each detailed step
    loop_steps = [key for key in list(etch_grid.keys()) if 'init' not in key]
    curr_process = 'init'
    d = cell_size
    diag_dist = np.sqrt(2*cell_size**2)
    
    run_accounting = False
    
    amounts = []
    
    etch_comp_time = []
    cell_accounting_time = []
    
    n_cells_span = 2
    
    start_time = time.time()
    
    save_states = {step: np.zeros_like(states) for i,step in enumerate(KOI)}
    save_phis = {step: np.zeros_like(phis) for i,step in enumerate(KOI)}
    
    
    for step_i, step in enumerate(loop_steps,start=1):
    
        #### plot every X steps
        if step_i % 2 == 0:#(int(len(loop_steps)/20)) == 0:
            # print('\tmaking plot for step %s' % step)
            path = plot_state_dir + step + '.png'
            utils.plot_cell_list(exposed_indices, coords,
                               phis=phis,step=step,save_to_file=path,
                               cell_size=cell_size)
            path = plot_z_dir + step + '.png'
            utils.plot_cell_list(exposed_indices, coords,
                               phis=None,step=step,save_to_file=path,
                               cell_size=cell_size)
    
        #### define current process
        master_step = step.split('_')[0]
        prev_process = curr_process
        if 'bosch-iso' in step:
            if 'isotime0' in step:
                curr_process = 'bosch-iso: bosch'
            else:
                curr_process = 'bosch-iso: iso'
        else:
            if 'bosch' in step.split('_')[1]:
                curr_process = 'bosch'
            elif 'iso' in  step.split('_')[1]:
                curr_process = 'iso'
        if curr_process != prev_process:
            print('current process: %s'
                  %(curr_process))
    
    
        etch_start_time = time.time()
        update_phis = np.zeros_like(phis,dtype=np.float32)
    
    
        #### bosch steps
        if (curr_process == 'bosch-iso: bosch' or curr_process == 'bosch'):
            # bosch step is a vertical etch of exposed_cells in the x, y bounds
            # of the mask ('in_mask' == True)
            n_bosch_steps = recipe_steps[master_step]['bosch']
            if curr_process == 'bosch-iso: bosch':
                curr_bosch_step = int(step.split('_')[-2].split('bosch')[-1])
            else:
                curr_bosch_step = int(step.split('_')[-1].split('bosch')[-1])
    
    
            print('\tbosch step %i of %i (step %i of %i)' % \
                  (curr_bosch_step, n_bosch_steps, step_i,n_steps))
    
            exp_pts = np.array([0,0,0])#make_cloud([exposed_cells])[0]
    
            for index in exposed_indices:
                i, j, k = index[0], index[1], index[2]
    
                if tuple([i,j]) in known_in_mask_indices:
                    z = coords[2][i,j,k]
                    etch_amount = bosch_vert_step(z) * \
                        known_in_mask_indices[tuple([i,j])]
                    update_phis[i,j,k] = etch_amount
    
    
    
    
        #### iso steps
        elif (curr_process == 'bosch-iso: iso' or curr_process == 'iso'):
    
            n_iso_steps = recipe_steps[master_step]['iso']
            curr_iso_step = int(step.split('_')[-1].split('isotime')[-1])
    
            print('\tiso time %i of %i seconds (step %i of %i)' % \
                  (curr_iso_step, n_iso_steps, step_i, n_steps))
    
            exp_pts = np.array([x_mid,y_mid,0])#make_cloud([exposed_cells])[0]
    
            angles = []
            x_center, y_center, z_center = [],[],[]
            dN = 0.01
            ea = 0
    
            for index in exposed_indices:
                i, j, k = index[0], index[1], index[2]
                phi_i_j_k = phis[i,j,k]
                normal = norms[i,j,k]
    
                # sweep the cell centers and get phis if available
                # e.g., ii for i+1 and 0 for i-1
    
                x, y, z = coords[0][i,j,k], coords[1][i,j,k], coords[2][i,j,k]
    
    
                i_forward_diff = (phis[i+1,j,k] - phi_i_j_k)/cell_size
                i_backward_diff = (phi_i_j_k - phis[i-1,j,k])/cell_size
    
                j_forward_diff = (phis[i,j+1,k] - phi_i_j_k)/cell_size
                j_backward_diff = (phi_i_j_k - phis[i,j-1,k])/cell_size
    
                k_forward_diff = (phis[i,j,k+1] - phi_i_j_k)/cell_size
                k_backward_diff = (phi_i_j_k - phis[i,j,k-1])/cell_size
    
                angle = utils.compute_angle(normal,ref_pt=exp_pts)
    
                curr_vert_rate = vert_rate(z)
                curr_horiz_rate = horiz_rate(z)
                # etch_rate
                etch_rate = np.sqrt((curr_vert_rate*np.cos(angle))**2 + \
                    (curr_horiz_rate*np.sin(angle))**2)
    
                beta_x_pos = 0.5*(i_forward_diff + i_backward_diff)
                beta_y_pos = 0.5*(j_forward_diff + j_backward_diff)
                beta_z_pos = 0.5*(k_forward_diff + k_backward_diff)
                grad_mag = np.sqrt(beta_x_pos**2 + beta_y_pos**2 + beta_z_pos**2)
    
                beta_x_neg = 0.5*(i_forward_diff - i_backward_diff)
                beta_y_neg = 0.5*(j_forward_diff - j_backward_diff)
                beta_z_neg = 0.5*(k_forward_diff - k_backward_diff)
    
                alpha_l = abs(np.array([curr_horiz_rate*normal[0],
                                    curr_horiz_rate*normal[1],
                                    curr_vert_rate*normal[2]]))
    
                etch_amount = C_etch*t_step*(etch_rate*grad_mag - \
                    alpha_f*(alpha_l[0]*beta_x_neg + alpha_l[1]*beta_y_neg + \
                          alpha_l[2]*beta_z_neg))
    
                ea += etch_amount/len(exposed_indices)
    
                update_phis[i,j,k] = etch_amount
    
        # remove etch amount for either active process
        phis -= update_phis
    
        dummy_phis = phis + 0.5*cell_size
        if dummy_phis.min() < 0:
            run_accounting = True
    
        start_accounting_time = time.time()
    
        # temp lists for exposed, removed, and neighbor cells
        temp_exposed = []
        temp_removed = []
        temp_neigh = []
    
        if run_accounting == True:
            run_accounting = False
    
            #### checking exposed cells to move to temporary containers
            dummy_exposed = []
            for index in exposed_indices:
                i, j, k = index[0], index[1], index[2]
                if (phis[i,j,k] < -0.5*cell_size):
                    temp_removed.append(index)
                elif (phis[i,j,k] > 0.5*cell_size):
                    temp_neigh.append(index)
                else:
                    dummy_exposed.append(index)
    
    
            #### adjust phi values for neighbor cells
            dummy_neigh = []
            for index in neigh_indices:
                i,j,k = index[0], index[1], index[2]
                nidx = utils.list_neighbor_indices(index)
                min_idx, min_phi = None, np.inf
                for idx in nidx:
                    i_n,j_n,k_n = idx[0], idx[1], idx[2]
                    if tuple(idx) in exposed_indices_set:
                        if phis[i_n,j_n,k_n] < min_phi:
                            min_phi = phis[i_n,j_n,k_n]
                            min_idx = idx
    
                if min_idx is None:
                    pass
                else:
                    phis[i,j,k] = min_phi + cell_size
                    if abs(phis[i,j,k]) <= 0.5*cell_size:
                        temp_exposed.append(index)
                    else:
                        dummy_neigh.append(index)
    
    
            #### adjust phi values for removed cells
            dummy_rm = []
            for index in rm_indices:
                i,j,k = index[0], index[1], index[2]
                nidx = utils.list_neighbor_indices(index)
                max_idx, max_phi = None, -np.inf
                for idx in nidx:
                    i_n,j_n,k_n = idx[0], idx[1], idx[2]
                    if tuple(idx) in exposed_indices_set:
                        if phis[i_n,j_n,k_n] > max_phi:
                            max_phi = phis[i_n,j_n,k_n]
                            max_idx = idx
    
                if max_idx is None:
                    pass
                else:
                    phis[i,j,k] = max_phi - cell_size
                    if abs(phis[i,j,k]) <= 0.5*cell_size:
                        temp_exposed.append(index)
                    else:
                        dummy_rm.append(index)
    
    
            # refresh exposed indices list and set for quick searching
            exposed_indices = dummy_exposed
            exposed_indices_set = set((i[0],i[1],i[2]) for i in exposed_indices)
    
            # refresh neighbor indices list and set for quick searching
            neigh_indices = dummy_neigh
            neigh_indices_set = set((i[0],i[1],i[2]) for i in neigh_indices)
            print('\t\t%i neighbor cells' % len(neigh_indices))
    
            # refresh removed indices list
            rm_indices = dummy_rm
            rm_indices_set = set((i[0],i[1],i[2]) for i in rm_indices)
            print('\t\t%i removed cells' % len(rm_indices))
    
    
            #### convert temp lists to main lists and refresh sets
            # neighs
            for index in temp_neigh:
                if tuple(index) not in neigh_indices_set:
                    i,j,k = index[0], index[1], index[2]
                    neigh_indices.append(index)
                    states[i,j,k] = 1
            neigh_indices_set = set((i[0],i[1],i[2]) for i in neigh_indices)
            # removed
            for index in temp_removed:
                if tuple(index) not in rm_indices_set:
                    i,j,k = index[0], index[1], index[2]
                    rm_indices.append(index)
                    states[i,j,k] = -1
            rm_indices_set = set((i[0],i[1],i[2]) for i in rm_indices)
            # exposed
            for index in temp_exposed:
                if tuple(index) not in exposed_indices_set:
                    i,j,k = index[0], index[1], index[2]
                    exposed_indices.append(index)
                    states[i,j,k] = 0
            exposed_indices_set = set((i[0],i[1],i[2]) for i in exposed_indices)
            print('\t\t%i exposed cells' % len(exposed_indices))
    
            # assign states exposed cell neighbors and refresh normals
            for index in exposed_indices:
                i,j,k = index[0], index[1], index[2]
                nidx = utils.list_neighbor_indices(index)
                for idx in nidx:
                    i_n,j_n,k_n = idx[0], idx[1], idx[2]
                    if states[i_n,j_n,k_n] == 2 and k_n < kk[-1]:
    
                        states[i_n,j_n,k_n] = 1
                        phis[i_n,j_n,k_n] = phis[i,j,k] + cell_size
    
                        if tuple(idx) not in neigh_indices_set:
                            neigh_indices.append(idx)
    
                    if states[i_n,j_n,k_n] == -2:
    
                        states[i_n,j_n,k_n] = -1
                        phis[i_n,j_n,k_n] = phis[i,j,k] - cell_size
    
                        if tuple(idx) not in rm_indices_set:
                            rm_indices.append(idx)
    
        #### update normals
        if step_i%50001 == 0:
            print('\t\t\tupdating all normals')
            norms = utils.compute_normals(states,coords,norms,
                                          exposed_indices,
                                          cell_size=cell_size)
        else:
            norms = utils.compute_normals(states, coords, norms, temp_exposed,
                                          cell_size)
    
        #### smooth signed distances (phi)
        if step_i%50 == 0:
            print('\t\t\tsmoothing phi')
            smooth_start = time.time()
    
            w = 0.5
            dummy = phis
            for index in exposed_indices:
                w_avg = 0
                i,j,k = index[0],index[1],index[2]
                dist,idx = utils.get_exposed_neighbors(index, states, coords,
                                                        n_cells_span=1)
                # dist,idx = utils.get_neighbors(index, states, coords,
                #                                        n_cells_span=1)
                n_neigh = len(idx)
                for i_d,neigh_index in enumerate(idx) :
                    i_n,j_n,k_n = neigh_index[0], neigh_index[1], neigh_index[2]
                    w_avg += (1-w)/n_neigh * phis[i_n,j_n,k_n]
                    # w_avg += phis[i_n,j_n,k_n]
    
                dummy[i,j,k] = w_avg+w*phis[i,j,k]  #(w_avg + phis[i,j,k])/(n_neigh+1)
                # dummy[i,j,k] = (w_avg + phis[i,j,k])/(n_neigh+1)
    
            phis = dummy
    
        # elif step_i%50 == 0:
        #     print('\t\t\tsmoothing phi')
        #     smooth_start = time.time()
    
        #     w = 0.7
        #     dummy = phis
        #     for index in exposed_indices:
        #         w_avg = 0
        #         i,j,k = index[0],index[1],index[2]
        #         # dist,idx = utils.get_exposed_neighbors(index, states, coords,
        #         #                                         n_cells_span=1)
        #         dist,idx = utils.get_neighbors(index, states, coords,
        #                                                n_cells_span=1)
        #         n_neigh = len(idx)
        #         for i_d,neigh_index in enumerate(idx) :
        #             i_n,j_n,k_n = neigh_index[0], neigh_index[1], neigh_index[2]
        #             w_avg += (1-w)/n_neigh * phis[i_n,j_n,k_n]
        #             # w_avg += phis[i_n,j_n,k_n]
    
        #         dummy[i,j,k] = w_avg+w*phis[i,j,k]  #(w_avg + phis[i,j,k])/(n_neigh+1)
        #         # dummy[i,j,k] = (w_avg + phis[i,j,k])/(n_neigh+1)
    
        #     phis = dummy
    
    
    
    
            # w = 0.5
            # phi_p_i = np.roll(phis, 1, axis=0)
            # phi_m_i = np.roll(phis, -1, axis=0)
            # phi_p_j = np.roll(phis, 1, axis=1)
            # phi_m_j = np.roll(phis, -1, axis=1)
            # phi_p_k = np.roll(phis, 1, axis=2)
            # phi_m_k = np.roll(phis, -1, axis=2)
            # phis = w * phis + (w-1)/6*(phi_p_i + phi_m_i + phi_p_j + \
            #                            phi_m_j + phi_p_k + phi_m_k)
    
            # ndimage.convolve(phis,kern, mode='constant', cval=0.0)
    
    
            # X, Y = coords[0][:,:,0], coords[1][:,:,0]
            # Z = np.ones_like(X) * phis[-1,-1,-1]
            # X_i, Y_i, Z_i = np.where(states == 0)
            # for i,j,k in zip(X_i,Y_i,Z_i):
            #     if states[i,j,k] == 0:
            #         Z[i,j] = phis[i,j,k]
            # zi = sg.convolve2d(Z, kernel, mode='same', boundary='fill',
            #                    fillvalue=-10)
            # for i,j,k in zip(X_i,Y_i,Z_i):
            #     if states[i,j,k] == 0:
            #         phis[i,j,k] = zi[i,j]
    
            print('\t\t\t\tsmoothing time %.2f s' %
                  (time.time()-smooth_start))
    
    
        accounting_time = time.time() - start_accounting_time
        print('\t\t\taccounting time %.2f s \n' % (accounting_time))
    
    
        #### save the etch profile
        try:
            save_states[step] = states.copy()
            save_phis[step] = phis.copy()
        except:
            pass
    
    
    print('-------- %.2f seconds ---------' % (time.time()-start_time))
    
    # %% take slice of etch
    for step in loop_steps:
        s_diag, z_diag = utils.cross_section_slice(phis, states, coords,
                                                   cell_size, normal=[1,1,0],
                                                   offset=0)
        s, z = utils.cross_section_slice(phis, states, coords,
                                         cell_size, normal=[1,0,0],
                                         offset=0)
    plt.plot(s_diag, z_diag,'o')
    plt.plot(s, z)
    
    # %% dummy
    from scipy.interpolate import griddata
    import scipy.signal as sg
    
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = coords[0][:,:,0], coords[1][:,:,0]
    Z = np.ones_like(X) * -10
    ii,jj = X.shape
    kk = coords[0].shape[2]
    for i in range(ii):
        for j in range (jj):
            for k in range(kk):
                if states[i,j,k] == 0:
                    Z[i,j] = phis[i,j,k]
                else:
                    pass
    ax.plot_surface(X,Y,Z)
    
    zi = griddata((X.ravel(), Y.ravel()), Z.ravel(),
                  (x_axis[None,:], y_axis[:,None]), method='nearest')
    
    X, Y = np.meshgrid(x_axis,y_axis)
    kernel_shape = (3,3)
    kernel = np.full(kernel_shape, 1/np.prod(kernel_shape))
    
    zi = sg.convolve2d(Z, kernel, mode='same', boundary='fill',
                       fillvalue=-10)
    zi = np.reshape(zi,X.shape)
    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y,zi)
    
    fig = plt.figure()
    
    plt.subplot(121)
    plt.imshow(zi)
    plt.subplot(122)
    plt.imshow(Z)
    
    
    
    # %% save files and plot final etch profile
    pv.set_plot_theme("document")
    written_keys = [key for key in save_phis.keys() if np.max(save_phis[key]) != 0]
    plot_states = save_states['step01_bosch-iso03_bosch012_isotime24']
    exposed_cells = utils.plot_pts(plot_states,coords,cell_size=cell_size,
                               wafer_thickness=525,
                               state='exposed',plot_grid=False)
    # exposed_pts,exposed_states,_ = utils.make_cloud([exposed_cells])
    #plot_point_cloud((exposed_pts,neigh_pts),scalar='z')
    
    # with_data = [g for g in etch_grid if len(etch_grid[g]) != 0]
    
    
    # dict_file = plot_z_dir + 'exposed_cells_maskX_R5_C3.txt'
    # # save exposed_cell to file
    # with open(dict_file, 'w') as file:
    #     file.write(json.dumps(exposed_cells))
    
    # #save vtk file
    for key in save_states.keys():
        plot_states = save_states[key]
        exposed_obj = pv.PolyData(utils.get_cell_coords(plot_states,coords,state = 'exposed'))
        vtk_save_exp_obj = plot_z_dir +key+ 'exposed_obj.vtk' 
    
        exposed_obj.save(vtk_save_exp_obj)
    
    #pcd_exp = o3d.geometry.PointCloud()
    #pcd_neigh = o3d.geometry.PointCloud()
    #pcd_exp.points = o3d.utility.Vector3dVector(exposed_pts)
    #pcd_neigh.points = o3d.utility.Vector3dVector(neigh_pts)
    #o3d.visualization.draw_geometries([pcd_exp,pcd_neigh])
    
    # steps = []
    # for step in list(etch_grid.keys()):
    #     if len(etch_grid[step]) != 0:
    #         steps.append(step)
    # last_key = steps[-50]#list(etch_grid.keys())[-1]
    # exp_pts = etch_grid[last_key][0]
    
    # norms = utils.compute_normals(states,coords,norms,
    #                                   exposed_indices,
    #                                   cell_size=cell_size)
    
    # exp_norms = []
    # exp_pts = []
    # for index in exposed_indices:
    #     i,j,k = index[0], index[1], index[2]
    #     exp_pts.append([coords[0][i,j,k], coords[1][i,j,k], coords[2][i,j,k]])
    #     exp_norms.append(norms[i,j,k])
    # pcd_exp = o3d.geometry.PointCloud()
    # pcd_exp.points = o3d.utility.Vector3dVector(exp_pts)
    # for norm in exp_norms:
    #     pcd_exp.normals.append(-norm)
    # o3d.visualization.draw_geometries([pcd_exp])
    
    
    # # %% get metrics out of final profile
    # # num_exp_n = []
    # # num_n = []
    # # angle = []
    # # z = []
    # # for cell in list(exposed_cells.keys()):
    # #     z.append(cell[2])
    # #     x = cell[0]
    # #     y = cell[1]
    # #     angle.append(np.arctan(y/x))
    # #     num_exp_n.append(len(exposed_cells[cell]['exp_neighbors']))
    # #     num_n.append(len(exposed_cells[cell]['neighbors']))
    # #     # print(exposed_cells[cell]['num_exp_neighs'] -
    # #     #       len(exposed_cells[cell]['neighbors']))
    
    # # plt.hist(num_exp_n)
    
    # # from mpl_toolkits.mplot3d import Axes3D
    # # import matplotlib.cm as cmx
    # # import matplotlib as mpl
    
    # # fig = plt.figure()
    # # ax = Axes3D(fig)
    # # cm = 'inferno'
    # # cNorm = mpl.colors.Normalize(vmin=0, vmax=max(num_n))
    # # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    # # ax.scatter(angle,num_exp_n,z,alpha=1,c=scalarMap.to_rgba(num_n),
    # #                     marker='o')
    # # fig.colorbar(scalarMap)
    # # ax.set_xlabel('angle')
    # # ax.set_ylabel('num exp neighbors')
    # # ax.set_zlabel('z')
    
    # # # # suffle y pos and gap values carrying their indices with them
    # # # temp = list(zip(i_y_pos,y_pos))
    # # # temp_yp = sorted(temp_yp,key=lambda temp_yp:temp_yp[1], reverse=True)
    # # # random.shuffle(temp_yp)
    # # # i_y_pos,y_pos = zip(*temp_yp)
    # # # i_y_pos = list(i_y_pos)
    # # # y_pos = list(y_pos)
    
    # # plt.figure()
    # # plt.plot(num_n,num_exp_n,'o')
    # # plt.xlabel('num exposed neighbors')
    # # plt.ylabel('total num neighbors')





