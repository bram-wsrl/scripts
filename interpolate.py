from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


contours_T = list[dict]


def add_q_pts(new_idx: np.array, q_short: np.array, q_long: np.array):
    '''
    Calculate representing points on other contour
    Extremes are excludes and always matched.
    '''
    q_new = q_long[new_idx]
    ql_src = q_long[new_idx[0] - 1]
    qr_src = q_long[-1]
    ql_dst = q_short[-2]
    qr_dst = q_short[-1]

    q_rel = (q_new - ql_src) / (qr_src - ql_src)
    return ql_dst + (qr_dst - ql_dst) * q_rel


def scaling(
        tt: int, ttmin: np.array, ttmax: np.array,
        min_axis: np.array, max_axis: np.array
        ):
    return min_axis + (max_axis - min_axis) / (ttmax - ttmin) * (tt - ttmin)


def update_contours(contours: contours_T):
    '''
    Inplace update of contours

    This function prepares for scaling
    Contours are updated so that they have the same number
    of nodes as the contour with the highest number of nodes.
    '''
    idx_longest = np.argmax([len(c['discharge']) for c in contours])
    q_long = contours[idx_longest]['discharge']
    q_idx_long = np.array(range(len(q_long))[1:-1])

    for contour in contours:
        h_short = contour['head']
        q_short = contour['discharge']

        if len(q_long) > len(q_short):
            q_idx_short = np.array(range(len(q_short))[1:-1])
            q_idx_new = np.setdiff1d(q_idx_long, q_idx_short)

            q_insert = add_q_pts(q_idx_new, q_short, q_long)
            h_insert = np.interp(q_insert, q_short, h_short)

            contour['discharge'] = np.insert(q_short, q_idx_new[0], q_insert)
            contour['head'] = np.insert(h_short, q_idx_new[0], h_insert)


def add_contours(contours: contours_T, dtt: int):
    '''
    Inplace addition of contours at dtt intervalls
    '''
    contours.sort(key=lambda x: x['speed'])
    preset_speeds = np.array([c['speed'] for c in contours])

    tt_min_global = contours[0]['speed']
    tt_max_global = contours[-1]['speed']
    tt_range = np.arange(tt_min_global, tt_max_global, dtt)[1:]

    temps = []
    for tt in tt_range:
        if tt in preset_speeds:
            continue

        insert_idx = np.searchsorted(preset_speeds, tt)
        contour_min = contours[insert_idx-1]
        contour_max = contours[insert_idx]

        tt_min = contour_min['speed']
        tt_max = contour_max['speed']

        q_min = contour_min['discharge']
        q_max = contour_max['discharge']

        h_min = contour_min['head']
        h_max = contour_max['head']

        q_scaling = scaling(tt, tt_min, tt_max, q_min, q_max)
        h_scaling = scaling(tt, tt_min, tt_max, h_min, h_max)
        temps.append({'speed': tt, 'head': h_scaling, 'discharge': q_scaling})
    contours.extend(temps)


def finetune_contours(contours: contours_T):
    '''
    Returns new contours list of dictionaries
    
    This function loops over all contour dictionairies in contours:
    All positive discharges are copied into new dictionairies.
    The first negative discharge is mapped to a discharge of zero and included
    into the new dictionaries.
    The other negative discharges are excluded.    
    '''
    contours_finetuned = []
    
    for contour in contours:
        index_first_negative_discharge = None
        for discharge_value in contour['discharge']:
            if discharge_value < 0:
                index_first_negative_discharge = list(contour['discharge']).index(discharge_value)
                break               
        
        if index_first_negative_discharge is None:
            head_list = contour["head"]
            discharge_list = contour["discharge"]
        else:
            head_list = contour["head"][:index_first_negative_discharge+1]
            discharge_list = contour["discharge"][:index_first_negative_discharge+1]
            discharge_list[index_first_negative_discharge] = 0
            
        contours_finetuned.append({
            "speed":contour["speed"],
            "head":head_list,
            "discharge":discharge_list
            })    
 
    return contours_finetuned


def plot(contours: contours_T, figpath: Path):
    fig, ax = plt.subplots()
    for contour in contours:
        ax.plot(contour['discharge'], contour['head'], label=contour['speed'], marker='x')
    ax.set_xlabel('discharge')
    ax.set_ylabel('head')
    ax.set_title(figpath.stem)
    ax.legend()
    fig.savefig(figpath)
