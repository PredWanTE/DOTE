NO_CONFIG = "No config file present, using default values."
FILE_MISSING = "The config file path %s does not exist, are you sure this is the right one?"
DEFAULT_FALLBACK = "Falling back to default properties file"

IGNORE_MSG = "Ignoring."

PARSING_CONFIG = "Config file found. Parsing: %s"
BAD_CONFIG_LINE = "Config line %d is not well formed, need 3 values but got %d"
BAD_CONFIG_LINE_IGNORE = BAD_CONFIG_LINE + ", "  + IGNORE_MSG 
BAD_TYPE_LINE = "Iine %d is malformed, got key:%s  -- value: %s -- type: %s"
BAD_TYPE_LINE_IGNORE = BAD_TYPE_LINE +", " +IGNORE_MSG
BAD_LIST_TYPE= "using list requires to have list,<type> -- we got: %s"
BAD_LIST_VALUE= "You have specified a list type of \"%s\"-- but we got: %s"

DOWNLOAD_METRICS_MSG = "Will download these metrics: %s"
WARN_METRIC_MISSING_FORCING = "[vmid=%s] Metric %s is missing, will force download metrics: %s"
FAILURE_HEADER_MSG = "We had a total of %d metrics failures across %d VMs, did we use forced download other metrics if did not get an exact match?: %s"
FAILED_MATRIC_STATS_MSG = "%s: failed on %f of the VMs"
ZERO_VALUES_OF_TOTAL = "%s: zero value percentage = %f"
NO_VALUES_OF_TOTAL = "%s: no values found during evaluation"