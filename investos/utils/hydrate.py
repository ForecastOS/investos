import os
import pickle

import investos as inv


class HydrateMixin:
    _excluded_attrs = {"strategy", "benchmark", "risk_free"}

    def _implied_attributes(self):
        props = {
            name
            for name in dir(type(self))
            if isinstance(getattr(type(self), name), property)
        }
        return self._excluded_attrs.union(props)

    def __getstate__(self):
        clean_state = {}
        self.__version__ = inv.__version__
        skip_keys = self._implied_attributes()

        for k, v in self.__dict__.items():
            if k in skip_keys:
                continue
            try:
                pickle.dumps(v)
                clean_state[k] = v
            except Exception as e:
                print(f"⚠️ Skipping non-pickleable attribute: {k} ({type(v)}): {e}")

        return clean_state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def dehydrate_to_disk(self, path: str, obj_name: str = "object.pkl"):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, obj_name), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def rehydrate_from_disk(cls, path: str, obj_name: str = "object.pkl"):
        with open(os.path.join(path, obj_name), "rb") as f:
            return pickle.load(f)
