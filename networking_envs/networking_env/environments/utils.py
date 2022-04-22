from networking_env.utils.shared_consts import SizeConsts

def norm_func(x, norm_val=1.*SizeConsts.ONE_Gb):
        return x/norm_val

def unnorm_func(x,norm_val=1.*SizeConsts.ONE_Gb):
        return x*norm_val
