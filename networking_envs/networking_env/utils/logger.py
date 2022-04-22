import datetime
import json
import os
import sys

import os.path as osp


LOG_OUTPUT_FORMATS = ['stdout', 'log']

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

NO_LOG = 50

level_to_name = {DEBUG: "DEBUG" , 
                INFO: "INFO",
                WARN: "WARNING",
                ERROR: "ERROR"} 

class OutputFormat(object):
    def writekvs(self, kvs):
        """
        Write key-value pairs
        """
        raise NotImplementedError

    def writeseq(self, args):
        """
        Write a sequence of other data (e.g. a logging message)
        """
        pass

    def close(self):
        return

    def init(self):
        return

class HumanOutputFormat(OutputFormat):
    def __init__(self, file):
        self.file = file

    def init(self):
        if type(self.file) == str:
            self.file = open(self.file, 'w')

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if isinstance(val, float):
                valstr = '%-8.3g' % (val,)
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        keywidth = max(map(len, key2str.keys()))
        valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = '-' * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items()):
            lines.append('| %s%s | %s%s |' % (
                key,
                ' ' * (keywidth - len(key)),
                val,
                ' ' * (valwidth - len(val)),
            ))
        lines.append(dashes)
        self.file.write('\n'.join(lines) + '\n')

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        return s[:50] + '...' if len(s) > 53 else s

    def writeseq(self, args):
        for arg in args:
            self.file.write(arg)
        self.file.write('\n')
        self.file.flush()

class JSONOutputFormat(OutputFormat):
    def __init__(self, file):
        self.file = file

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, 'dtype'):
                v = v.tolist()
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + '\n')
        self.file.flush()

def make_output_format(log_frmt, log_dir, fname="log.log"):
    os.makedirs(log_dir, exist_ok=True)
    if log_frmt == 'stdout':
        return HumanOutputFormat(sys.stdout)
    elif log_frmt == 'log':
        log_file = open(osp.join(log_dir, fname), 'wt')
        return HumanOutputFormat(log_file)
    elif log_frmt == 'json':
        json_file = open(osp.join(log_dir, fname), 'wt')
        return JSONOutputFormat(json_file)
    else:
        raise ValueError('Unknown format specified: %s' % (log_frmt,))


# ================================================================
# API
# ================================================================

def log_key_value(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    """
    get_logger().logkv(key, val)

def log_key_values(d):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        log_key_value(k, v)

def dump():
    """
    Write all of the diagnostics from the current iteration
    level: int. (see logger.py docs) If the global logger level is higher than
                the level argument here, don't print to stdout.
    """
    get_logger().dumpkvs()

def get_key_values():
    return get_logger().name2val    
 
def log(*args, level=INFO):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    get_logger().log(*args, level=level)
  
  
def debug(*args):
    log(*args, level=DEBUG)
  
  
def info(*args):
    log(*args, level=INFO)
  
  
def warn(*args):
    log(*args, level=WARN)
  
  
def error(*args):
    log(*args, level=ERROR)
  
  
def set_level(level):
    """
    Set logging threshold on current logger.
    """
    get_logger().set_level(level)
  
def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return get_logger().get_dir()
 
 
def configure(log_dir=None, format_strs=None):
        assert Logger.CURRENT is Logger.DEFAULT,\
            "Only call logger.configure() when it's in the default state. Try calling logger.reset() first."
        assert log_dir is not None, "Please specify a log directory before calling logger"
         
        if log_dir is not None:
            log_dir = get_log_dir(log_dir)
             
        if format_strs is None:
            format_strs = LOG_OUTPUT_FORMATS
        output_formats = [make_output_format(f, log_dir) for f in format_strs]
        Logger.CURRENT = Logger(log_dir=log_dir, output_formats=output_formats)
         
        info('Logging to %s'%dir)
#  

# ================================================================
# Backend
# ================================================================

class Logger(object):
    DEFAULT = None  # A logger with no output files. (See right below class definition)
                    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, log_dir, output_formats):
        # make sure that the log folder actually exists
        os.makedirs(log_dir, exist_ok=True)
        
        self.name2val = {}  # values this iteration
        self.level = INFO
        self.dir = log_dir
        self.output_formats = output_formats
        
        #init output formats
        for fmt in self.output_formats:
            fmt.init()
        

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val):
        self.name2val[key] = val

    def dumpkvs(self):
        if self.level == NO_LOG: return
        for fmt in self.output_formats:
            fmt.writekvs(self.name2val)
        self.name2val.clear()

    def log(self, *args, level=INFO):
        args_ = [ level_to_name[level]+": "]  +list(args)
        if self.level <= level:
            self._do_log(args_)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        self.level = level

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            fmt.writeseq(args)



# ================================================================
# Set logger default value
# ================================================================
def get_log_dir(log_dir="/tmp/"):
    return osp.join(log_dir,  datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) #-%S-%f 

def get_log_file(log_dir="/tmp/"):
    return osp.join(log_dir,  "logger.log")


Logger.DEFAULT = Logger.CURRENT = Logger(log_dir=get_log_dir(), 
                                         output_formats=[HumanOutputFormat(sys.stdout)]  )

# ================================================================
# General logging operations
# ================================================================ 
def get_logger():
    return Logger.CURRENT      
 
def reset():
        Logger.CURRENT = Logger.DEFAULT
        info('Reset logger')
    
log_tabular = log_key_value