from networking_env.utils import logger, log_msgs
from networking_env.utils.globals import PropertiesConsts

class ConfigFile(object):
    '''
    This class contains a generic configuration file
    '''
    CURRENT = None

    def __init__(self, fname=None, strict=False):
        self._parse(fname, strict)

    def __str__(self):
        header = "*"*5 + " Config File " + "*"*5
        res = header + "\n"
        for key in self.__dict__:
            res += " "*5 + "%s\t:\t%s\n"%(str(key), str(self.__dict__[key]))
        res += "*"*len(header) + "\n"
        return res

    def _update(self, args, force_update = True):
        for key in args:
            if force_update or key in self.__dict__:
                self.__dict__[key] = args[key]

    def _do_parse(self, fname, strict=True, sep=PropertiesConsts.LINE_SEP):
        """
        Assuming each line is formatted like so:
        key    value    type
        So a tuple separated by a tab (\t) letter
        
        type could be either:
            *  list,<type> - in that case values are assumed to be comma delimited
                all list values are assumed to be of the same type
                e.g.:

            * str - value is considered as string
            
            * int - value is converted to int
            
            * float - value is converted to float
        """
        with open(fname, 'r') as f:
            lines = f.readlines()
            lines_ = [l.split(sep) for l in lines]
            
            lines = []
            for lid, line in enumerate(lines_):
                if len(line) != 3:
                    if strict:
                        raise ValueError( log_msgs.BAD_CONFIG_LINE%(lid, len(line)) )
                    else:
                        logger.warn(log_msgs.BAD_CONFIG_LINE_IGNORE%(lid, len(line) ) )
                
                lines.append( (lid, line) )
            
            values = {}
            
            for ind, data in lines:
                k, v, t = list(map(lambda x: x.strip(), data))
                
                if t == PropertiesConsts.INT_TYPE:
                    values[k] = int(v)
                elif t == PropertiesConsts.FLOAT_TYPE:
                    values[k] = float(v)
                elif t == PropertiesConsts.STR_TYPE:
                    values[k] = v
                elif PropertiesConsts.LIST_SUB_TYPE in t:
                    values[k] = ConfigFile.parse_list(v, t)
                else:
                    if strict:
                        raise ValueError( log_msgs.BAD_TYPE_LINE%(ind, k,v,t) ) 
                    else:
                        logger.warn( log_msgs.BAD_TYPE_LINE_IGNORE%(ind, k,v,t) )
            
            self._update(values)

    @staticmethod
    def parse_list(v_org, t, sep=PropertiesConsts.LIST_SEP):
        try:
            t = t.split(sep)
            assert len(t) == 2
            t = t[1]
        except:
            raise ValueError( log_msgs.BAD_LIST_TYPE%t )
        
        try:
            v = v_org.split(sep)
            if t == PropertiesConsts.STR_TYPE:
                return v
            elif t == PropertiesConsts.INT_TYPE:
                return list( map( int,v ) )
            elif t == PropertiesConsts.FLOAT_TYPE:
                return list( map( float,v ) )
        except:
            raise ValueError( log_msgs.BAD_LIST_VALUE%(t, v_org ) )

    def _parse(self, fname, strict):
        """
        Parse the config file
        """
        import os
        if fname == None or fname == "": 
            logger.info(log_msgs.NO_CONFIG)
            return
        
        if strict:
            assert os.path.exists(fname), log_msgs.FILE_MISSING%fname
            self._do_parse(fname, strict)
             
        else:
            if not os.path.exists(fname):
                logger.warn(log_msgs.FILE_MISSING%fname)
                logger.warn(log_msgs.DEFAULT_FALLBACK)
            else:
                logger.info(log_msgs.PARSING_CONFIG%fname) 
                self._do_parse(fname, strict)


ConfigFile.CURRENT = ConfigFile()

def get_data_dir():
    return get_config().data_base +"/"+ get_config().data_dir + "/"

def get_config():
    return ConfigFile.CURRENT

def update(fname, is_strict):
    ConfigFile.CURRENT = ConfigFile(fname,is_strict)
    
def reset():
    ConfigFile.CURRENT = ConfigFile()

def dump_config(base_folder):
    with open(base_folder+"/config.conf", 'w') as f:
        f.write(str(get_config()))
