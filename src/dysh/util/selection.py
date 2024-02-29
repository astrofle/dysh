import numbers
import warnings
from collections.abc import Sequence
from copy import deepcopy

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import Angle
from astropy.table import Table  # , TableAttribute
from astropy.units.quantity import Quantity
from pandas import DataFrame

# from ..fits import default_sdfits_columns
from . import generate_tag

default_aliases = {
    "freq": "crval1",
    "ra": "crval2",
    "dec": "crval3",
    "glon": "crval2",
    "glat": "crval3",
    "gallon": "crval2",
    "gallat": "crval3",
    "elevation": "elevatio",
    "source": "object",
    "pol": "plnum",
}


class Selection(DataFrame):
    """This class contains the methods for selecting data from an SDFITS object.
    Data (rows) can be selected using any column name in the input SDFITS object.
    Exact selection, range selection, upper/lower limit selection, any-of selection
    are all supported.
    """

    def __init__(self, sdfits, aliases=default_aliases, **kwargs):
        super().__init__(sdfits._index, copy=True)
        warnings.simplefilter("ignore", category=UserWarning)
        idtag = ["ID", "TAG"]
        # if we want Selection to replace _index in sdfits
        # construction this will have to change. if hasattr("_index") etc
        DEFKEYS = list(sdfits._index.keys())
        # add ID and TAG as the first columns
        for i in range(len(idtag)):
            DEFKEYS.insert(i, idtag[i])
        # remove bintable
        DEFKEYS.remove("BINTABLE")
        DEFKEYS = np.array(DEFKEYS)
        dt = np.array([str] * len(DEFKEYS))
        # add number selected column which is an int
        DEFKEYS = np.insert(DEFKEYS, len(DEFKEYS), "# SELECTED")
        dt = np.insert(dt, len(dt), np.int32)
        # ID is also an int
        dt[0] = np.int32
        self._table = Table(data=None, names=DEFKEYS, dtype=dt)
        for t in idtag:
            self._table.add_index(t)
        self._valid_coordinates = ["RA", "DEC", "GALLON", "GALLAT", "GLON", "GLAT", "CRVAL2", "CRVAL3"]
        self._selection_rules = dict()
        self._aliases = dict()
        self.alias(**aliases)
        warnings.resetwarnings()

    @property
    def aliases(self):
        """
        The aliases that may be used to refer to SDFITS columns,
        Returns
        -------
        dict
            The dictionary of aliases and SDFITS column names

        """
        return self._aliases

    def alias(self, **aliases):
        """
        Alias a set of keywords to existing columns. Multiple aliases for
        a single column are allowed, e.g.,
        { 'glon':'crval2', 'lon':'crval2'}

        Parameters
        ----------
        aliases : dict()
            The dictionary of keywords and column names
            where the new alias is the key and
            the column name is the value and , i.e., {alias:column}

        Returns
        -------
        None.

        Raises
        ------
            ValueError if the column name is not recognized.
        """
        self._check_keys(aliases.values())
        for k, v in aliases.items():
            self._alias(k, v)

    def _alias(self, key, column):
        """
        Alias a new keyword to an existing column, e.g..
        to alias the SDFITS column 'CRVAL2' as 'RA':

            `alias('RA','CRVAL2')`

        The map is case insensitive, so `alias('ra', 'crval2')` also works.

        Parameters
        ----------
        key : str
            The new keyword to use as an alias.
        column : str
            The existing SDFITS column name to alias

        Returns
        -------
        None.

        """
        self._aliases[key.upper()] = column.upper()

    def _set_pprint_exclude_names(self):
        """Use `~astropy.Table.pprint_exclude_names` to set the list
        columns that have no entries.
        """
        if len(self._table) > 0:
            emptycols = np.array(self._table.colnames)[
                [np.all([self._table[k].data[i] == "" for i in range(len(self._table))]) for k in self._table.colnames]
            ]
            self._table.pprint_exclude_names.set(emptycols)

    def _sanitize_input(self, key, value):
        """
        Sanitize a key-value pair for. Coordinate types are checked for.

        Parameters
        ----------
        key : str
            upper case key value

        value : any
            The value for the key

        Returns
        -------
        sanitized_value : str
            The sanitized value
        """
        # @todo   Allow minimum match str for key
        if key in self._aliases.keys():
            key = self._aliases[key]
        if key not in self:
            raise KeyError(f"{key} is not a recognized column name.")
        v = self._sanitize_coordinates(key, value)
        self._check_for_disallowed_chars(key, value)
        return v

    def _sanitize_coordinates(self, key, value):
        """
        Sanitize a coordinate selection key-value pair. Coordinates will be
        converted to floats before the final value is created.

        Parameters
        ----------
        key : str
            upper case key value

        value : any
            The value for the key.  It can be a single float,
            a single Angle (Quantity), a tuple of Angles
            (a1,a2,a3) or an Angle tuple, e.g., (n1,n2)*u.degree

        Returns
        -------
        sanitized_value : str
            The sanitized value.
        """
        if key not in self._valid_coordinates and key not in self.aliases:
            return value
        # note Quantity is derivative of np.ndarray, so
        # need to filter that out in the recursive call.
        # This is to handle (q1,q2) as a range.
        # (n1,n2)*u.degree is handled below
        if isinstance(value, (tuple, np.ndarray, list)) and not isinstance(value, Quantity):
            return [self._sanitize_coordinates(key, v) for v in value]
        if isinstance(value, numbers.Number):
            a = Angle(value * u.degree)
        else:  # it should be a str or Quantity
            a = Angle(value)
        return a.degree

    def _check_for_disallowed_chars(self, key, value):
        # are there any?  coordinates will already
        # be transformed to decimal degrees
        pass

    def _generate_tag(self, values, hashlen=9):
        """
        Generate a unique tag based on row values.  A hash object is
        created from the input values using SHA256, and a hex representation is created.
        The first `hashlen` characters of the hex string are returned.

        Parameters
        ----------
        values : array-like
            The values to use in creating the hash object
        hashlen : int, optional
            The length of the returned hash string. The default is 9.

        Returns
        -------
        tag : str
            The hash string

        """
        return generate_tag(values, hashlen)

    @property
    def _next_id(self) -> int:
        """
        Get the next ID number in the table.

        Returns
        -------
        id : int
            The highest existing ID number plus one
        """
        ls = len(self._table)
        if ls == 0:
            return 0
        return max(self._table["ID"]) + 1

    def _check_keys(self, keys):
        """
        Check a dictionary for unrecognized keywords.  This method is called in any select method to check inputs.

        Parameters
        ----------
        keys : list or array-like
           Keyword arguments

        Returns
        -------
        None.

        Raises
        ------
        KeyError
            If one or more keywords are unrecognized

        """
        unrecognized = []
        ku = [k.upper() for k in keys]
        for k in ku:
            if k not in self and k not in self._aliases:
                unrecognized.append(k)
        # print("KU, K", ku, k)
        if len(unrecognized) > 0:
            raise KeyError(f"The following keywords were not recognized: {unrecognized}")

    def _check_numbers(self, **kwargs):
        self._check_type(numbers.Number, "Expected numeric value for these keywords but did not get a number", **kwargs)

    def _check_range(self, **kwargs):
        bad = []
        for k, v in kwargs.items():
            ku = k.upper()
            if not isinstance(v, (tuple, list, np.ndarray)):
                raise ValueError(f"Invalid input for key {ku}={v}. Range inputs must be tuple or list.")
            for a in v:
                if a is not None:
                    if isinstance(a, Quantity):
                        a = self._sanitize_coordinates(ku, a)
                    try:
                        self._check_numbers(**{ku: a})
                    except ValueError:
                        bad.append(ku)
        if len(bad) > 0:
            raise ValueError(f"Expected numeric value for these keywords but did not get a number: {bad}")

    def _check_type(self, reqtype, msg, **kwargs):
        # @todo allow Quantities
        """
        Check that a list of keyword arguments is all a specified type.

        Parameters
        ----------

        reqtype : type
            The object type to check against, e.g. numbers.Number, str, etc

        msg : str
            The exception message to show if the inputs are not the specific reqtype

        **kwargs : dict or key=value
           Keyword arguments

        Raises
        ------
        ValueError
            If one or more of the values is not numeric.

        Returns
        -------
        None.

        """
        ku = np.ma.masked_array([k.upper() for k in kwargs.keys()])
        ku.mask = np.array([isinstance(x, reqtype) for x in kwargs.values()])
        if not np.all(ku.mask):
            raise ValueError(f"{msg}: {ku[~ku.mask]}")

    def _check_for_duplicates(self, df):
        """
        Check that the user hasn't already added a rule matching this one

        Parameters
        ----------
        df : ~pandas.DataFrame
            The selection to check

        Raises
        ------
        Exception
            If an identical rule (DataFrame) has already been added.

        Returns
        -------
        None.

        """
        for _id, s in self._selection_rules.items():
            if s.equals(df):
                # print(s, df)
                tag = self._table.loc[_id]["TAG"]
                raise Exception(f"A rule that results in an identical has already been added. ID: {_id}, TAG:{tag}")

    def _addrow(self, row, dataframe, tag=None):
        """
        Common code to add a tagged row to the internal table after the selection has been created.
        Should be called in select* methods.

        Parameters
        ----------
        tag : str, optional
            An identifying tag by which the rule may be referred to later.
            If None, a  randomly generated tag will be created.
        row : dict
            key, value pairs of the selection
        dataframe : ~pandas.DataFrame
            The dataframe created by the selection.

        Returns
        -------
        None.

        """
        self._check_for_duplicates(dataframe)
        if tag is not None:
            row["TAG"] = tag
        else:
            row["TAG"] = self._generate_tag(list(row.values()))
        row["ID"] = self._next_id
        row["# SELECTED"] = len(dataframe)
        self._selection_rules[row["ID"]] = dataframe
        self._table.add_row(row)

    def select(self, tag=None, **kwargs):
        """Add one or more exact selection rules, e.g., `key1 = value1, key2 = value2, ...`
        If `value` is array-like then a match to any of the array members will be selected.
        For instance `select(object=['3C273', 'NGC1234']) will select data for either of those
        objects and `select(ifnum=[0,2])` will select IF number 0 or IF number 2.

        Parameters
        ----------
            tag : str
                An identifying tag by which the rule may be referred to later.
                If None, a  randomly generated tag will be created.
            key : str
                The key  (SDFITS column name or other supported key)
            value : any
                The value to select

        """
        self._check_keys(kwargs.keys())
        row = dict()
        df = self
        for k, v in list(kwargs.items()):
            ku = k.upper()
            if ku in self._aliases:
                ku = self._aliases[ku]
            v = self._sanitize_input(ku, v)
            # If a list is passed in, it must be composed of strings.
            # Numeric lists are intepreted as ranges, so must be
            # selected by user with select_range
            if isinstance(v, (Sequence, np.ndarray)) and not isinstance(v, str):
                print(ku, v)
                query = None
                for vv in v:
                    # self._check_type(
                    #     str,
                    #    "Numeric arrays are not allowed with exact selection, use select_range instead.",
                    #    **{ku: vv},
                    # )
                    # if it is a string, then OR them.
                    # e.g. object = ["NGC123", "NGC234"]
                    if isinstance(vv, str):
                        thisq = f'{ku} == "{vv}"'
                    else:
                        thisq = f"{ku} == {vv}"
                    if query is None:
                        query = thisq
                    else:
                        query += f"| {thisq}"
                    # for pd.merge to give the correct answer, we would
                    # need "inner" on the first one and "outer" on subsequent
                    # df = pd.merge(df, df[df[ku] == vv], how="inner")
                # print("final query ", query)
                df = df.query(query)
            else:
                df = pd.merge(df, df[df[ku] == v], how="inner")
            row[ku] = str(v)
        if df.empty:
            warnings.warn("Your selection rule resulted in no data being selected. Ignoring.")
            return
        self._addrow(row, df, tag)
        # return df

    def select_range(self, tag=None, **kwargs):
        """
        Select a range of inclusive values for a given key(s).
        e.g., `key1 = (v1,v2), key2 = (v3,v4), ...`
        Will select data  `v1 <= data1 <= v2, v3 <= data2 <= v4, ... `
        Upper and lower limits may be given by setting one of the tuple values
        to None. e.g., `key1 = (None,v1)` for an upper limit `data1 <= v1` and
        `key1 = (v1,None)` for a lower limit `data >=v1`.  Lower
        limits may also be specified by a one-element tuple `key1 = (v1,)`.

        Parameters
        ----------
        tag : str, optional
            An identifying tag by which the rule may be referred to later.
            If None, a  randomly generated tag will be created.
        key : str
            The key (SDFITS column name or other supported key)
        value : array-like
            Tuple or list giving the lower and upper limits of the range.

        Returns
        -------
        None.

        """
        self._check_keys(kwargs.keys())
        self._check_range(**kwargs)
        row = dict()
        df = self
        for k, v in list(kwargs.items()):
            ku = k.upper()
            if ku in self._aliases:
                ku = self._aliases[ku]
            v = self._sanitize_input(ku, v)
            print(f"{ku}={v}")
            # deal with a tuple quantity
            if isinstance(v, Quantity):
                v = v.value
            vn = list()
            # deal with quantity inside a tuple.
            for q in v:
                # ultimately will need a map of
                # desired units, so e.g. if
                # GHz used, then the value is expressed in Hz
                if isinstance(q, Quantity):
                    vn.append(q.value)
                else:
                    vn.append(q)
            v = vn
            row[ku] = str(v)
            if len(v) == 2:
                if v[0] is not None and v[1] is not None:
                    df = pd.merge(df, df[(df[ku] <= v[1]) & (df[ku] >= v[0])], how="inner")
                elif v[0] is None:  # upper limit given
                    df = pd.merge(df, df[(df[ku] <= v[1])], how="inner")
                else:  # lower limit given (v[1] is None)
                    df = pd.merge(df, df[(df[ku] >= v[0])], how="inner")
            elif len(v) == 1:  # lower limit given
                df = pd.merge(df, df[(df[ku] >= v[0])], how="inner")
            else:
                raise Exception(f"Couldn't parse value tuple {v} for key {k} as a range.")
        if df.empty:
            warnings.warn("Your selection rule resulted in no data being selected. Ignoring.")
            return
        self._addrow(row, df, tag)

    def select_within(self, tag=None, **kwargs):
        """
        Select a value within a plus or minus for a given key(s).
        e.g. `key1 = [value1,epsilon1], key2 = [value2,epsilon2], ...`
        Will select data
        `value1-epsilon1 <= data1 <= value1+epsilon1,`
        `value2-epsilon2 <= data2 <= value2+epsilon2,...`

        Parameters
        ----------
        tag : str, optional
            An identifying tag by which the rule may be referred to later.
            If None, a  randomly generated tag will be created.
        key : str
            The key (SDFITS column name or other supported key)
        value : array-like
            Tuple or list giving the value and epsilon

        Returns
        -------
        None.

        """
        # This is just a type of range selection.
        kw = dict()
        for k, v in kwargs.items():
            v1 = v[0] - v[1]
            v2 = v[0] + v[1]
            kw[k] = (v1, v2)
        self.select_range(tag, **kw)

    def select_channel(self, chan):
        """
        Select channels and/or channel ranges. These are NOT used in final()
        but rather will be used to create a mask for calibration or
        flagging.

        Parameters
        ----------
        chan : number, or array-like
            The channels to select

        Returns
        -------
        None.

        """
        pass

    def select_time(self, time):
        """
        Select time(s)

        Parameters
        ----------
        time : probably an astropy.Time object
        or something that can be converted to it

            The time(s) to select

        Returns
        -------
        None.

        """
        pass

    def remove(self, id=None, tag=None):
        """Remove (delete) a selection rule(s).
        You must specify either `id` or `tag` but not both. If there are
        multiple rules with the same tag, they will all be deleted.

        Parameters
        ----------
            id : int
                The ID number of the rule as displayed in `show()`
            tag : str
                An identifying tag by which the rule may be referred to later.
        """
        if id is not None and tag is not None:
            raise Exception("You can only specify one of id or tag")
        if id is None and tag is None:
            raise Exception("You must specify either id or tag")
        if id is not None:
            if id in self._selection_rules:
                # We will assume that selection_rules and table
                # have been kept in sync.  The implementation
                # should ensure this.
                del self._selection_rules[id]
                row = self._table.loc_indices["ID", id]
                # there is only one row per ID
                self._table.remove_row(row)
            else:
                raise KeyError(f"No ID = {id} found in this Selection")
        else:
            # need to find IDs of selection rules where TAG == tag.

            # This will raise keyerror if tag not matched, so no need
            # to raise our own, unless we want to change the messgae.
            matching_indices = self._table.loc_indices["TAG", tag]
            #   raise KeyError(f"No TAG = {tag} found in this Selection")
            matching = self._table[matching_indices]
            self._table.remove_rows(matching_indices)

            for i in matching["ID"]:
                del self._selection_rules[i]
                # self._selection_rules.pop(i, None) # also works

    # This method commented out until further notice.
    # It has issues with multiple ways of removing table rows
    # creating an inconsistent table
    # def clear(self):
    #   """
    #    Remove all selection rules
    #
    #    Returns
    ##    -------
    #    None.
    #
    #    """
    #   self._selection_rules = {}
    # remove[0] will fail if the user has already
    # removed it, which then screws up the index and no further
    # modification becomes possible
    #   self._table.remove_rows([0, len(self._table) - 1])

    def show(self):
        """
        Print the current selection rules. Only columns with a rule are shown.
        The first two columns are ID number a TAG string. Either of these may be used
        to :meth:remove a row.  The final column `# SELECTED` gives
        the number of rows that a given rule selects from the original.
        The :meth:final selection may be fewer rows because each selection rule
        is logically OR'ed to create the final selection.

        Returns
        -------
        None.

        """
        self._set_pprint_exclude_names()
        print(self._table)

    @property
    def final(self):
        """
        Create the final selection. This is done by a logical OR of each
        of the selection rules (specifically `pandas.merge(how='inner')`).

        Returns
        -------
        final : DataFrame
            The resultant selection from all the rules.
        """
        # start with unfiltered index.
        # make a copy to avoid reference to self
        final = deepcopy(self)
        for df in self._selection_rules.values():
            final = pd.merge(final, df, how="inner")
        return final
