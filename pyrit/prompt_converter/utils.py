import json

try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    check_if_streamlit_running = get_script_run_ctx
except ImportError:
    check_if_streamlit_running = lambda: False

def mock_cache(func=None, *args, **kwargs):
    if func is None:
        return lambda f: f
    return func

# Check if we are in a streamlit app: if so, expose caches
# https://discuss.streamlit.io/t/check-if-running-in-a-streamlit-session/842/6
if check_if_streamlit_running(): # We are running an app with `streamlit run`
    st_cache_data = st.cache_data
    st_cache_resource = st.cache_resource
else: # Running a standard python script; disable caches
    st_cache_data =  mock_cache
    st_cache_resource = mock_cache


def read_json(fpath):
    with open(fpath) as infile:
        return json.load(infile)


@st_cache_data(show_spinner=False)
def write_json(obj, fpath):
    with open(fpath, 'w') as outfile:
        json.dump(obj, outfile, indent=2)


@st_cache_data(show_spinner=False)
def write_csv(df, fpath, **kwargs):
    df.to_csv(fpath, **kwargs)