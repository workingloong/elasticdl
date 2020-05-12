"""This util gets statistics from the environment and return a default value
if there is no statistics in the environment. So user can set a default value
to perform unit tests.
"""
import os


class AnalysisEnvTemplate:
    MIN_ENV = "_{}_min"
    MAX_ENV = "_{}_max"
    AVG_ENV = "_{}_avg"
    STDDEV_ENV = "_{}_stddev"
    BUCKET_BOUNDARIES_ENV = "_{}_boundaries"
    DISTINCT_COUNT_ENV = "_{}_distinct_count"
    VOCABULARY_ENV = "_{}_vocab"


def get_min(feature_name, default_value):
    """Get the min value of numeric feature from the environment.
    Return the default value if there is no the statistics in
    the environment.

    Args:
        feature_name: String, feature name or column name in a table
        default_value: Float.

    Return:
        Float
    """
    env_name = AnalysisEnvTemplate.MIN_ENV.format(feature_name)
    min_value = os.getenv(env_name, None)
    if min_value is None:
        return default_value
    else:
        return float(min_value)


def get_max(feature_name, default_value):
    """Get the max value of numeric feature from the environment.
    Return the default value if there is no the statistics in
    the environment.

    Args:
        feature_name: String, feature name or column name in a table
        default_value: Float.

    Return:
        Float
    """
    env_name = AnalysisEnvTemplate.MAX_ENV.format(feature_name)
    max_value = os.getenv(env_name, None)
    if max_value is None:
        return default_value
    else:
        return float(max_value)


def get_avg(feature_name, default_value):
    """Get the average of numeric feature from the environment.
    Return the default value if there is no the statistics in
    the environment.

    Args:
        feature_name: String, feature name or column name in a table
        default_value: Float.

    Return:
        Float
    """
    env_name = AnalysisEnvTemplate.AVG_ENV.format(feature_name)
    mean = os.getenv(env_name, None)
    if mean is None:
        return default_value
    else:
        return float(mean)


def get_stddev(feature_name, default_value):
    """Get the standard deviation from the environment.
    Return the default value if there is no the statistics in
    the environment.

    Args:
        feature_name: String, feature name or column name in a table
        default_value: Float.

    Return:
        Float.
    """
    env_name = AnalysisEnvTemplate.STDDEV_ENV.format(feature_name)
    std_dev = os.getenv(env_name, None)
    if std_dev is None:
        return default_value
    else:
        return float(std_dev)


def get_bucket_boundaries(feature_name, default_value):
    """Get the bucket boundaries from the environment.
    Return the default value if there is no the statistics in
    the environment.

    Args:
        feature_name: String, feature name or column name in a table
        default_value: List with float values.

    Return:
        List with float values.
    """
    env_name = AnalysisEnvTemplate.BUCKET_BOUNDARIES_ENV.format(feature_name)
    boundaries = os.getenv(env_name, None)
    if boundaries is None:
        return default_value
    else:
        boundaries = list(map(float, boundaries.split(",")))
        return sorted(set(boundaries))


def get_distinct_count(feature_name, default_value):
    """Get the count of distinct feature values set from the environment.
    Return the default value if there is no the statistics in
    the environment.

    Args:
        feature_name: String, feature name or column name in a table
        default_value: Integer.

    Return:
        Integer.
    """
    env_name = AnalysisEnvTemplate.DISTINCT_COUNT_ENV.format(feature_name)
    count = os.getenv(env_name, None)
    if count is None:
        return default_value
    else:
        return int(count)


def get_vocabulary(feature_name, default_value):
    """Get the feature vocabulary from the environment.
    Return the default value if there is no the statistics in
    the environment.

    Args:
        feature_name: String, feature name or column name in a table
        default_value: List with strings or a path of vocabulary files.

    Return:
        List with strings.
    """
    env_name = AnalysisEnvTemplate.VOCABULARY_ENV.format(feature_name)
    vocabulary_path = os.getenv(env_name, None)
    if vocabulary_path is None:
        return default_value
    else:
        return vocabulary_path
