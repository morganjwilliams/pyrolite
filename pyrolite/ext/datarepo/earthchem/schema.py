"""
Generating schemas for Earthchem-Sourced data.

1. Infer data types from tables
2. Integrate with Earthchem XML schema
"""
# %% --
import xmltodict

# from http://www.earthchemportal.org/schema/earthchem_schema.xsd
with open("earthchem_schema.xsd") as fd:
    doc = xmltodict.parse(fd.read())

schema = doc["xs:schema"]
schema.keys()
schema["@version"]
# %% -


class Dataset(object):
    def __init__(self):
        pass

    def build(self):
        pass


class Field(object):
    def __init__(self, parent=None):
        pass


class Metadata(Field):
    def __init__(self):
        super().__init__()


class ID(Metadata):
    def __init__(self):
        super().__init__()


class Location(Metadata):
    def __init__(self):
        pass


class Latitude(Location):
    def __init__(self):
        super().__init__()


class Longitude(Location):
    def __init__(self):
        super().__init__()


class CompositionalVariable(Field):
    def __init__(self):
        super().__init__()


class IsotopeRatio(Field):
    def __init__(self):
        super().__init__


class InitialIsotopeRatio(IsotopeRatio):
    def __init__(self):
        super().__init__()


class DeltaIsotopeRatio(IsotopeRatio):
    def __init__(self):
        super().__init__()


class EpsilonIsotopeRatio(IsotopeRatio):
    def __init__(self):
        super().__init__()


class Activity(Field):
    def __init__(self):
        super().__init__()


class GasConcentration(Field):
    def __init__(self):
        super().__init__()
