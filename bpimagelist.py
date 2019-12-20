import pandas as pd
import numpy as np
import mmap
from copy import copy

img = [
    'rec_header',
    'client',
    'date1',
    'date2',
    'version',
    'bp_image_id',
    'policy',
    'policy_type',
    'proxy_client',
    'creator',
    'schedule',
    'sched_type',
    'ret_lvl',
    'backup_time',
    'duration_sec',
    'expire_time',
    'compression',
    'encryption',
    'size_kb',
    'file_n',
    'copy_n',
    'frag_n',
    'f_file_compressed',
    'f_file',
    'sw_ver',
    'name1',
    'in_opts',
    'primary_copy',
    'image_type',
    'tir',
    'tir_expire',
    'key_word',
    'mpx',
    'extended_sec',
    'file_from_raw',
    'dump_lvl',
    'fs_only',
    'last_incr',
    'last_full',
    'description',
    'req_id',
    'status_code',
    'backup_copy_type',
    'prev_bp_image',
    'job_id',
    'n_resumes',
    'resume_expire',
    'f_file_size',
    'pfu_type',
    'image_attr',
]
img65 = [
    'class_id',
    'slp_name',
    'slp_done',
    'snap_time',
]
img655 = [
    'snap_ver',
]
img71 = [
    'remote_expire',
    'orig_master',
    'origin_master_guid',
]
img75 = [
    'ir_enabled',
    'charset',
    'on_hold',
    'indexing_status',
]

img_dtypes = {
    'rec_header': np.object,
    'client': np.object,
    'date1': np.dtype(np.int),
    'date2': np.dtype(np.int),
    'version': np.dtype(np.int),
    'bp_image_id': np.object,
    'policy': np.object,
    'policy_type': np.dtype(np.int),
    'proxy_client': np.object,
    'creator': np.object,
    'schedule': np.object,
    'sched_type': np.dtype(np.int),
    'ret_lvl': np.dtype(np.int),
    'backup_time': np.dtype(np.int),
    'duration_sec': np.dtype(np.int),
    'expire_time': np.dtype(np.int),
    'compression': np.dtype(np.int),
    'encryption': np.dtype(np.int),
    'size_kb': np.dtype(np.int),
    'num_files': np.dtype(np.int),
    'num_copies': np.dtype(np.int),
    'num_frag': np.dtype(np.int),
    'f_file_compressed': np.dtype(np.int),
    'f_file': np.object,
    'sw_ver': np.object,
    'name1': np.object,
    'in_opts': np.dtype(np.int),
    'primary_copy': np.dtype(np.int),
    'image_type': np.dtype(np.int),
    'tir': np.dtype(np.int),
    'tir_expire': np.dtype(np.int),
    'key_word': np.object,
    'mpx': np.dtype(np.int),
    'extended_sec': np.dtype(np.int),
    'file_from_raw': np.dtype(np.int),
    'last_incr': np.dtype(np.int),
    'last_full': np.dtype(np.int),
    'description': np.object,
    'req_id': np.dtype(np.int),
    'status_code': np.dtype(np.int),
    'prev_bp_image': np.object,
    'job_id': np.dtype(np.int),
    'n_resumes': np.dtype(np.int),
    'resume_expire': np.dtype(np.int),
    'f_file_size': np.dtype(np.int),
    'image_attr': np.dtype(np.int),
    'class_id': np.object,
    'slp_name': np.object,
    'slp_done': np.dtype(np.int),
    'snap_time': np.dtype(np.int),
    'snap_ver': np.dtype(np.int),
    'remote_expire': np.dtype(np.int),
    'orig_master': np.object,
    'origin_master_guid': np.object,
    'ir_enabled': np.dtype(np.int),
    'charset': np.dtype(np.int),
    'on_hold': np.dtype(np.int),
    'indexing_status': np.dtype(np.int),
}

frag = [
    'hdr',
    'copy_n',
    'frag_n',
    'size_kb',
    'remainder_b',
    'media_type',
    'density',
    'file_n',
    'media_id',
    'host',
    'block_size_b',
    'offset',
    'media_time',
    'device',
    'f_flags',
    'media_descr',
    'expire_time',
    'mpx',
    'ret_lvl',
    'checkpoint',
    'resume_nbr',
    'media_seq',
    'media_subtype',
    'try_to_keep_time',
    'copy_time',
    'unused1',
]
frag_652 = ['kms_key']
frag_655 = ['dest_slp_name']
frag_75 = [
    'frag_state',
    'data_format',
    'kms_key',
    'dest_slp_name',
    'mirror_parent_copy_n',
    'on_hold'
]

frag_dtypes = {
    'hdr': np.object,
    'copy_n': np.dtype(np.int),
    'frag_n': np.dtype(np.int),
    'size_kb': np.dtype(np.int),
    'remainder_b': np.dtype(np.int),
    'media_type': np.dtype(np.int),
    'density': np.dtype(np.int),
    'file_n': np.dtype(np.int),
    'media_id': np.object,
    'host': np.object,
    'block_size_b': np.dtype(np.int),
    'offset': np.dtype(np.int),
    'media_time': np.dtype(np.int),
    'device': np.dtype(np.int),
    'f_flags': np.dtype(np.int),
    'media_descr': np.object,
    'expire_time': np.dtype(np.int),
    'mpx': np.dtype(np.int),
    'ret_lvl': np.dtype(np.int),
    'checkpoint': np.dtype(np.int),
    'resume_nbr': np.dtype(np.int),
    'media_seq': np.dtype(np.int),
    'media_subtype': np.dtype(np.int),
    'try_to_keep_time': np.dtype(np.int),
    'copy_time': np.dtype(np.int),
    'unused1': np.dtype(np.int),
    'frag_state': np.dtype(np.int),
    'data_format': np.dtype(np.int),
    'kms_key': np.object,
    'dest_slp_name': np.object,
    'mirror_parent_copy_n': np.dtype(np.int),
    'on_hold': np.dtype(np.int),
}


def parse_bpimagelist(nbu_version, path):
    img_field = copy(img)
    frag_field = copy(frag)
    if nbu_version >= '6.5':
        img_field += img65
    if '6.5.2' <= nbu_version < '7.5':
        frag_field += frag_652
    if '6.5.5' <= nbu_version < '7.5':
        frag_field += frag_655
    if nbu_version >= '6.5.5':
        img_field += img655
    if nbu_version >= '7.1':
        img_field += img71
    if nbu_version >= '7.5':
        frag_field = frag_field[:-1] + frag_75
        img_field += img75

    frag_field.append('bp_image_id')

    this_bp_imge_id = ''
    lines = 0
    img_l = []
    frag_l = []
    with open(path, 'r+b') as fh:
        mm = mmap.mmap(fh.fileno(), 0)
        for line in iter(mm.readline, b""):
            lines += 1
            if lines % 50000 == 0:
                print("Read [{}] lines".format(lines))
            parts = line.decode("ascii").split()
            if parts[0] == 'IMAGE':
                if len(parts) != len(img_field):
                    new_parts = []
                    sub_p = ''
                    is_inside = False
                    for p in parts:
                        if '"' in p:
                            sub_p += p
                            if is_inside:
                                new_parts.append(p)
                                sub_p = ''
                                is_inside = False
                            else:
                                is_inside = True
                        elif not is_inside:
                            new_parts.append(p)
                        else:
                            sub_p += p
                    parts = new_parts
                img_l.append(parts)
                this_bp_imge_id = parts[5]
            elif parts[0] == 'FRAG':
                parts.append(this_bp_imge_id)
                frag_l.append(parts)
    return pd.DataFrame(img_l, columns=img_field), pd.DataFrame(frag_l, columns=frag_field)
