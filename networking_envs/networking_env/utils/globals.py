class TimeConsts:

    @staticmethod
    def ms_to_hour( ms ):
        return TimeConsts.sec_to_min(TimeConsts.ms_to_sec(ms)) / 60.

    @staticmethod
    def sec_to_hour( secs):
        return TimeConsts.sec_to_min(secs) / 60.

    @staticmethod
    def min_to_hour( mins):
        return mins/60.

    @staticmethod
    def sec_to_min( sec):
        return sec/60.
    
    @staticmethod
    def ms_to_sec( ms):
        return ms*1e-3
    
    @staticmethod
    def ms_to_min( ms):
        return TimeConsts.ms_to_sec(ms)/60.
    
    @staticmethod
    def sec_to_ms( seconds):
        return seconds*1e3
    
    @staticmethod
    def min_to_sec( minutes):
        return minutes*60
    
    @staticmethod
    def hour_to_min( hours):
        return hours*60
    
    @staticmethod
    def hour_to_sec( hours):
        return TimeConsts.min_to_sec(TimeConsts.hour_to_min( hours))
    
    @staticmethod
    def min_to_ms( minutes):
        return TimeConsts.sec_to_ms(TimeConsts.min_to_sec( minutes))
    
    @staticmethod
    def hour_to_ms( hours ):
        return TimeConsts.sec_to_ms(TimeConsts.min_to_sec( TimeConsts.hour_to_min(hours) ) )

class PropertiesConsts:
    LINE_SEP = "\t"
    
    STR_TYPE = "str"
    INT_TYPE = "int"
    FLOAT_TYPE = "float"
    LIST_SUB_TYPE = "list"
   
def dump_to_file(fname, data):
    import dill
    with open(fname, 'wb') as f:
        dill.dump(data,f)
        
def load_from_file(fname):
    import dill
    with open(fname, 'rb') as f:
        res = dill.load(f)
    return res

def load_file_values(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
        times  = [float(l.split(",")[0]) for l in lines if l]
        values  = [float(l.split(",")[1]) for l in lines if l]
    return times,values

def write_file_values(fname, times, values):
    with open(fname, 'w') as f:
        for t,v in zip(times,values):
            f.write("%s,%s\n"%(str(t), str(v)))