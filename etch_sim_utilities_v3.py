import numpy as np
import pyvista as pv
import matplotlib as mpl
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor

# helper function s for cellular automata silicon ethcing simulation

def define_steps(recipe_steps, t_start, t_step):
    from collections import OrderedDict
    total_iso_time = 0
    etch_grid = OrderedDict()
    etch_grid['init'] = []
    print('constructing specific step keys for topo container')
    for step in recipe_steps:
        try:
            total_iso_time += recipe_steps[step]['iso']
        except:
            pass
        for i_cycle, cycles in enumerate(range(recipe_steps[step]['cycles'])):
            # if it is a combined bosch-iso step
            if len(str(i_cycle)) < 2:
                if i_cycle == 9:
                    i_cycle_str = str(i_cycle+1)
                else:
                    i_cycle_str = '0' + str(i_cycle+1)

            if recipe_steps[step]['bosch'] != None and \
            recipe_steps[step]['iso'] != None:
                # construct detailed keys for data container
                # i.e. key step1_bosch-iso6_bosch12_isotime100 is data for
                # the 100th second if an iso etch following the 12th bosch step
                # in the 6th cycle of a bosch-iso combined 1st step of the recipe
                # combined bosch-iso etching starts with a bosch step; the key
                # for this first step can be identified by an "_isotime0" flag
                for i_bosch in range(recipe_steps[step]['bosch']):
                    if len(str(i_bosch)) < 3:
                        if len(str(i_bosch)) == 1:
                            if i_bosch == 9:
                                i_bosch_str = '0' + str(i_bosch+1)
                            else:
                                i_bosch_str = '00' + str(i_bosch+1)
                        elif len(str(i_bosch)) == 2:
                            if i_bosch == 99:
                                i_bosch_str = str(i_bosch+1)
                            else:
                                i_bosch_str = '0' + str(i_bosch+1)

                    # initial bosch cycle key
                    key = step + '_bosch-iso' + i_cycle_str + \
                          '_bosch' + i_bosch_str + '_isotime0'
                    etch_grid[key] = []
                for i_t,t in enumerate(range(t_start,
                                             recipe_steps[step]['iso'],
                                             t_step)):
                    key = step + '_bosch-iso' + i_cycle_str + \
                          '_bosch' + i_bosch_str + '_isotime' + str(t+t_step)
                    etch_grid[key] = []
            elif recipe_steps[step]['bosch'] != None and \
            recipe_steps[step]['iso'] == None:
                # similar key construction but specifically for bosch etching; it
                # assumed that each cycle of bosch etching has the same etching
                # rate
                for i_bosch in range(recipe_steps[step]['bosch']):
                    if len(str(i_bosch)) < 3:
                        if len(str(i_bosch)) == 1:
                            if i_bosch == 9:
                                i_bosch_str = '0' + str(i_bosch+1)
                            else:
                                i_bosch_str = '00' + str(i_bosch+1)
                        elif len(str(i_bosch)) == 2:
                            if i_bosch == 99:
                                i_bosch_str = str(i_bosch+1)
                            else:
                                i_bosch_str = '0' + str(i_bosch+1)
                    else:
                        i_bosch_str = str(i_bosch+1)

                    key = step + '_bosch' + i_bosch_str
                    etch_grid[key] = []

            elif recipe_steps[step]['bosch'] == None and \
            recipe_steps[step]['iso'] != None:
                # similar key construction but specifically for iso etching; it is
                # possible to have multiple cycles of iso etching, i.e. each with
                # different conditions/rates
                for i_t,t in enumerate(range(t_start,
                                             recipe_steps[step]['iso'],
                                             t_step)):
                    key = step + '_iso' + i_cycle_str + '_isotime' + \
                          str(t+t_step)
                    etch_grid[key] = []

    return etch_grid, total_iso_time

def ion_source_dist(theta, sigma=1):
    J = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(theta-np.pi/2)**2/(2*sigma**2))
    return J

def etch_rate():
    k_b = 1.38e-23  # Boltzman constant [J/K]
    T_s = 100 + 274.15  # substrate temperature [K]
    k_0 = np.linspace(0,30,30)
    F_r = 150  # flow rate of SF6 [sccm]
    return

#def cross_section_slice(cells,p1=(-200,200),p2=(200,-200)):
#    if type(cells) == dict:
#        cells = np.array(list(cells.keys()))

def cross_section_slice(phis, states, coords, cell_size, normal, offset=0):

    p1 = [coords[0][0,0,0], coords[1][0,0,0]]
    p2 = [coords[0][-1,-1,0], coords[1][-1,-1,0]]

    x_slice = np.arange(p1[1],p2[1],cell_size)

    m = normal[1]/normal[0]

    if offset == 0: offset = (p2[1] - p2[0])/2
    y_slice = m*(x_slice-p1[0]) + p1[1] + offset

    x_axis = np.reshape(coords[0][0,:,0],(-1,1))
    y_axis = np.reshape(coords[1][:,0,0],(-1,1))

    kk = states.shape[2]

    z = []
    x_out = []

    pt0 = np.array([x_slice[0], y_slice[0]])

    for x,y in zip(x_slice,y_slice):
        i_x = np.argmin((np.abs(x_axis - x)))
        i_y = np.argmin((np.abs(y_axis - y)))
        i_z = None
        for k in range(kk):
            if states[i_x,i_y,k] == 0:
                i_z = k
        if i_z == None: i_z = kk-2

        z.append(coords[2][i_x,i_y,i_z])
        pt = np.array([x,y])
        x_out.append(np.sqrt(np.sum((pt-pt0)**2)))

    return(x_out,z)

def list_neighbor_indices(index):
    i,j,k = index[0], index[1], index[2]
    return [[i-1,j,k], [i+1,j,k], [i,j-1,k], [i,j+1,k], [i,j,k-1], [i,j,k+1]]

def min_dist_to_etch_front(index, exposed_indices, coords, mp):
    dist_path = np.inf
    i,j,k = index[0], index[1], index[2]
    coord = np.array([coords[0][i,j,k], coords[1][i,j,k],
                      coords[2][i,j,k]]).reshape((-1,3))

    front_x = np.array([coords[0][ind[0],ind[1],ind[2]]
                        for ind in exposed_indices])
    front_y = np.array([coords[1][ind[0],ind[1],ind[2]]
                        for ind in exposed_indices])
    front_z = np.array([coords[2][ind[0],ind[1],ind[2]]
                        for ind in exposed_indices])

    etch_coords = np.hstack((front_x.reshape((-1,1)),
                             front_y.reshape((-1,1)),
                             front_z.reshape((-1,1))))

    dist = (np.sqrt(np.sum((etch_coords - coord)**2, axis=1))).min()

    if mp is not None:
        dist_path = (np.sqrt(np.sum((mp - coord)**2,
                                    axis=1))).min()

    if dist < dist_path:
        return dist
    else:
        return dist_path



def is_in_mask(x,y,mask_paths,radius=0.0,alt_label=False):
    # True for x, y pt coord in mask contour
    # False for x, y pt coord not in mask contour
    # SurfBound for pt on wafer surface as the boundary of an etch

    try:
        # one point, multiple mask paths
        for path in mask_paths:
            if alt_label != False:
                inside = alt_label
                break
            elif alt_label == False:
                inside = mask_paths[path].contains_point((x,y),radius=radius)
                if inside == True: break
    except:
        # multiple points, one mask path
        if type(x) is np.ndarray:
            if alt_label == False:
                inside = [mask_paths.contains_point((i,j),radius=radius) \
                          for i,j in zip(x,y)]
            elif alt_label != True:
                inside = [alt_label for i,j in zip(x,y)]
        else:
            if alt_label == False:
                inside = mask_paths.contains_point((x,y),radius=radius)
            elif alt_label != False:
                inside = alt_label
    return inside

def get_exposed_neighbors(index, states, coords, n_cells_span=1):
    i,j,k = index[0], index[1], index[2]
    i_span = np.linspace(i-n_cells_span,
                         i+n_cells_span,
                         2*n_cells_span+1,dtype=np.int)
    j_span = np.linspace(j-n_cells_span,
                         j+n_cells_span,
                         2*n_cells_span+1,dtype=np.int)

    if k + n_cells_span >= states.shape[2]:
        n_cells_span = states.shape[2]-k-1

    k_span = np.linspace(k-n_cells_span,
                         k+n_cells_span,
                         2*n_cells_span+1,dtype=np.int)

    idx = []
    dist = []
    x,y,z = coords[0][i,j,k],coords[1][i,j,k],coords[2][i,j,k]
    for i_s in i_span:
        for j_s in j_span:
            for k_s in k_span:
                if states[i_s,j_s,k_s] == 0:
                    x_n,y_n,z_n = coords[0][i_s,j_s,k_s], \
                        coords[1][i_s,j_s,k_s],coords[2][i_s,j_s,k_s]

                    idx.append([i_s,j_s,k_s])
                    d = np.sqrt(((x_n-x)**2 + (y_n-y)**2 + (z_n-z)**2))
                    dist.append(d)

    return dist,idx

def get_neighbors(index, states, coords, n_cells_span=1):
    i,j,k = index[0], index[1], index[2]
    i_span = np.linspace(i-n_cells_span,
                         i+n_cells_span,
                         2*n_cells_span+1,dtype=np.int)
    j_span = np.linspace(j-n_cells_span,
                         j+n_cells_span,
                         2*n_cells_span+1,dtype=np.int)

    if k + n_cells_span >= states.shape[2]:
        n_cells_span = states.shape[2]-k-1

    k_span = np.linspace(k-n_cells_span,
                         k+n_cells_span,
                         2*n_cells_span+1,dtype=np.int)

    idx = []
    dist = []
    x,y,z = coords[0][i,j,k],coords[1][i,j,k],coords[2][i,j,k]
    for i_s in i_span:
        for j_s in j_span:
            for k_s in k_span:
                x_n,y_n,z_n = coords[0][i_s,j_s,k_s], \
                    coords[1][i_s,j_s,k_s],coords[2][i_s,j_s,k_s]

                idx.append([i_s,j_s,k_s])
                d = np.sqrt(((x_n-x)**2 + (y_n-y)**2 + (z_n-z)**2))
                dist.append(d)

    return dist,idx


def compute_normals(states, coords, norms,
                    etched_i_list, cell_size):
    import time
    # import itertools
    start_time = time.time()
    diag = cell_size * np.sqrt(3/4)
    n_cells_span = 3

    # radius = 7 * cell_size/2 + cell_size * 0.105  # 7 half-sizes plus margin away
    radius = (2*n_cells_span) * diag  # 6 diags plus margin away
    # radius = (cell_size/2) * np.sqrt(24) + (cell_size/2)*0.102
    n_normals = 0
    for index in etched_i_list:
        i,j,k = index[0], index[1], index[2]
        x,y,z = coords[0][i,j,k], coords[1][i,j,k], coords[2][i,j,k]

        n_normals += 1
        i_span = np.linspace(i-n_cells_span,
                             i+n_cells_span,
                             2*n_cells_span+1,dtype=np.int)
        j_span = np.linspace(j-n_cells_span,
                             j+n_cells_span,
                             2*n_cells_span+1,dtype=np.int)

        if k + n_cells_span >= states.shape[2]:
            n_cells_span = states.shape[2]-k-1
        k_span = np.linspace(k-n_cells_span,
                             k+n_cells_span,
                             2*n_cells_span+1,dtype=np.int)

        normal_vect = np.array([0,0,0])
        for i_s in i_span:
            for j_s in j_span:
                for k_s in k_span:
                    if states[i_s,j_s,k_s] < 0:
                        x_v = coords[0][i_s,j_s,k_s]
                        y_v = coords[1][i_s,j_s,k_s]
                        z_v = coords[2][i_s,j_s,k_s]
                        d = np.sqrt((x-x_v)**2 + (y-y_v)**2 + (z-z_v)**2)
                        if d < radius:
                            curr_vect = np.array([x_v - x,
                                                  y_v - y,
                                                  z_v - z])
                            mag = np.linalg.norm(curr_vect)
                            if mag != 0:
                                curr_unit_vect = curr_vect/mag
                            else:
                                curr_unit_vect = np.array([0,0,0])
                            normal_vect = np.add(normal_vect,
                                                 curr_unit_vect)
        if np.linalg.norm(normal_vect) == 0:
                # if this happens, it is a point in space with a spehere of
                # neighbors making the normal_vect magnitude 0 so just assign a
                # dummy normal
                norm = np.array([0,0,1])
        else:
            norm = (normal_vect / np.linalg.norm(normal_vect))

        norms[i,j,k] = norm

    t = time.time() - start_time
    print('\t\t\t\t%.2f s to assign normals to %i cells' % (t,n_normals))

    return norms

def compute_angle(norm,ref_pt=[0,0,0],wafer_thickness=525):
    ref_vec = np.array([ref_pt[0] - ref_pt[0],
                        ref_pt[1] - ref_pt[1],
                        wafer_thickness + 10000 - ref_pt[2]])
    ref_vec = np.array([0,0,1])# = ref_vec / np.linalg.norm(ref_vec)
    angle1 = np.arccos(np.dot(ref_vec,norm))
    angle2 = np.arccos(np.dot(ref_vec,-1*norm))
    if angle2 > angle1:
        angle = angle1
    else:
        angle = angle2
    return angle



def plot_cell_list(cell_list,coords,phis=None,
                   step=None,wafer_thickness=525,
                   save_to_file=False,cell_size=None):

    from pyvistaqt import BackgroundPlotter
    import pyvista as pv

    pts = []
    scalars = []

    for index in cell_list:
        i,j,k = index[0], index[1], index[2]
        pts.append([coords[0][i,j,k],
                    coords[1][i,j,k],
                    coords[2][i,j,k]])
        if phis is None:
            scalars.append(coords[2][i,j,k])
        else:
            scalars.append(phis[i,j,k])

    obj = pv.PolyData(pts)
    if save_to_file == False:
        plotter = BackgroundPlotter()
    else:
        plotter = pv.Plotter(off_screen=True)
    plotter.isometric_view()

    if phis is None:
        title = 'z height'
        clim = [min(obj.points[:,2]), wafer_thickness]
    else:
        title = 'signed distance'
        clim = [-cell_size/2, cell_size/2]

    plotter.isometric_view()
    plotter.add_mesh(obj, show_edges=True,
                     scalars=scalars,
                     point_size=5,
                     render_points_as_spheres=True,
                     cmap='Spectral',
                     clim=clim)

    plotter.add_scalar_bar(title=title, height=0.08,width=0.4,
                           position_x=0.01,position_y=0.01)
    if step != None:
        plotter.add_text(step)
    if save_to_file != False:
        file_name = save_to_file
        plotter.screenshot(file_name,transparent_background=True)
        plotter.close()

def plot_pts(plot_states,coords,cell_size,state,wafer_thickness=525,
             plot_grid=False):

    from pyvistaqt import BackgroundPlotter
    import pyvista as pv

    pts = get_cell_coords(plot_states,coords,state=state)
    if plot_grid is True:
        obj = make_grid(pts,cell_size)
    else:
        obj = pv.PolyData(pts)

    plotter = BackgroundPlotter()
    plotter.isometric_view()
    title = 'z height'
    clim = [min(obj.points[:,2]), wafer_thickness]

    if plot_grid is True:
        plotter.add_mesh(obj, show_edges=False,
                         scalars=obj.points[:,2],
                         point_size=cell_size,
                         render_points_as_spheres=True,
                         cmap='Spectral',
                         clim=clim)
    else:
        plotter.add_mesh(obj, show_edges=True,
                         scalars=obj.points[:,2],
                         point_size=cell_size,
                         render_points_as_spheres=True,
                         cmap='Spectral',
                         clim=clim)

    plotter.add_scalar_bar(title=title, height=0.08,width=0.4,
                           position_x=0.01,position_y=0.01)



def get_cell_coords(plot_states,coords,state):

    pts = []
    scalars = []

    if state == 'exposed':
        ii,jj,kk = np.where(plot_states==0)
    elif state == 'neighbors':
        ii,jj,kk = np.where(plot_states==1)
    for i,j,k in zip(ii,jj,kk):
        pts.append([coords[0][i,j,k],
                    coords[1][i,j,k],
                    coords[2][i,j,k]])
    return pts


def make_grid(cell_coords,cell_size):
    # helper function to make an unstructured grid from cell center points
    cells = []
    offset = np.arange(0,9*len(cell_coords),9)
    cell_type = np.array([pv.vtk.VTK_HEXAHEDRON]*len(cell_coords))
    point_count = 0
    pts = np.array([])
    n_pts = len(cell_coords)
    for i_cell, cell_center in enumerate(cell_coords):
        if i_cell%1000 == 0: print('building cell %i of %i' %(i_cell,n_pts))
        cells.append(8)
        for p in range(8):
            cells.append(point_count)
            point_count += 1
        x = cell_center[0]
        y = cell_center[1]
        z = cell_center[2]
        d = cell_size/2
        # note this order
        cell_pts = np.array([[x-d,y-d,z-d],[x+d,y-d,z-d],
                             [x+d,y+d,z-d],[x-d,y+d,z-d],
                             [x-d,y-d,z+d],[x+d,y-d,z+d],
                             [x+d,y+d,z+d],[x-d,y+d,z+d]])
        cell_pts = np.around(cell_pts,3)
        if pts.size == 0:
            pts = cell_pts
        else:
            pts = np.vstack((pts,cell_pts))
    # make unstructured grid
    grid = pv.UnstructuredGrid(offset,
                               np.array(cells),
                               cell_type,
                               pts)

    return grid

