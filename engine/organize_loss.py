def control_loss(Train_stage):
    if Train_stage == 'PoseNet_only':
        name_fs_list = ['Rot1', 'Rot2', 'Rot1_cos', 'Rot2_cos', 'Rot_regular', 'Tran', 'Size', 'R_con']
        name_recon_list = ['Per_point', 'Point_voting']
        name_geo_list = ['Geo_point']
        name_prop_list = ['Prop_pm', 'Prop_sym']
        name_TDA_list = []
    elif Train_stage == 'FSNet_only':
        name_fs_list = ['Rot1', 'Rot2', 'Tran', 'Size', 'Recon']
        name_recon_list = []
        name_geo_list = []
        name_prop_list = []
        name_TDA_list = []
    elif Train_stage == 'TDA':
        name_fs_list = []
        name_TDA_list = ['Rot1', 'Rot2', 'Rot1_cos', 'Rot2_cos', 'Rot_regular', 'Tran', 'Size', 'R_con', 'TDA_h1',
                         'TDA_h2', 'TDA_h1_cate', 'TDA_h2_cate', 'Prop_sym', 'R_DCD_cate_pred']
        name_recon_list = []
        name_geo_list = []
        name_prop_list = []
    else:
        raise NotImplementedError
    return name_fs_list, name_recon_list, name_geo_list, name_prop_list, name_TDA_list
