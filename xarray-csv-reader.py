import pandas as pd
import xarray as xr

def read_sweep_csv(path, skipcols=None, **kwargs):
    '''
    Given a path to a CSV file that has a carteian product structure
    will return a xarray of the file.
    
    This function looks at the unique number of values in each column to
    determine which features are coordinates and which are variables. We assume
    that the smallest number of unique values are coordinates and the largest
    number of unique values are variables (this may not be true if your
    model is not very dynamic and you sweep a large number of hyperparameter
    values).
    
    ----
    path: The file path
    
    skipcols: Name of any columns to skip while reading the CSV
    '''
    # Read CSV columns
    columns = pd.read_csv(path, nrows=0, **kwargs).columns
    
    # Create a set of columns to read from
    if (skipcols is not None):
        usecols = (set(columns) - set(skipcols))
    else:
        usecols = None
    # Read in the CSV as a dataframe
    df = pd.read_csv(path, usecols=usecols, **kwargs)
       
    # Save the count of unique values in each column
    value_counts = df.apply(lambda x: len(pd.unique(x))).sort_values()
    
    length = len(df)
    # Running product used to make sure number of 
    # coordinates multiply to length of data
    cumproduct = 1
    coord, var = [], []
    valid = False # Flag for product check
    
    # We will keep selecting columns working from least number of 
    # unique count to most number of unique counts
    for col, value in zip(value_counts.keys(), value_counts.values):
        # If there is only one unique value for a column we will save it 
        # in the meta data and skip using it as a coordinate
        if value == 1:
            continue
        # While we still have dims to add to the tensor add the column
        # with the next smallest number of unique values
        elif cumproduct < length:
            coord.append(col)
            cumproduct *= value
            # If the running product equals the length of the data 
            # clear the flag
            if cumproduct == length:
                valid = True
        # Add remaining columns to the variables  
        else:
            var.append(col)
    
    # Error message for invalid data shape
    if not valid:
        raise RuntimeError('Invalid data structure. Seems like you do not have'\
                    'the correct number of columns in your dataframe to create a cross product.'\
                    ' Try dropping a column with the skipcols argument')
        
    # Add any columns with only one unique value to the meta data
    meta_cols = (value_counts[value_counts == 1].keys())
    
    # Convert the dataframe to a xarray dataset using coords and vars
    ds = (df[coord+var].copy()
            .set_index(coord)
            .to_xarray()
            .assign_attrs(
                dict(zip(meta_cols, df.loc[0, meta_cols].values))
            )
         )
    
    return ds
    
    
def read_netlogo_table(path):
    '''
    Given a netLogo behavior space table CSV returns an xarray dataset 
    '''
    # Read skipping the run number (this is extra info)
    ds = read_sweep_csv(path, skipcols=['[run number]'], skiprows=6)
    
    # Adding meta data from first 6 rows of netlogo document
    df = pd.read_csv(path, skiprows=0, nrows=3)
    model_info = list(df.columns) + list(df.values.reshape(1, -1)[0])
    df = pd.read_csv(path, skiprows=4, nrows=1)
    model_info += [dict(zip(df.columns, df.values[0]))]
    
    # Add an additional entry to attrs under metadata
    ds.attrs['Metadata'] = model_info
    
    return ds