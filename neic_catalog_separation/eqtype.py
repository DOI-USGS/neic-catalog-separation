from enum import Enum

NO_VALUE = 99999


class EqType(Enum):
    """Enumerated values for types of earthquakes."""

    CRUSTAL = 1
    SUBDUCTION_INTERFACE = 2
    SUBDUCTION_INTRASLAB = 3
    SUBDUCTION_OUTERRISE = 4
    MANTLE = 5

    CRUSTAL_SUBDUCTION_INTERFACE = 12
    CRUSTAL_MANTLE = 15

    SUBDUCTION_INTERFACE_CRUSTAL = 21
    SUBDUCTION_INTERFACE_SUBDUCTION_INTRASLAB = 23
    SUBDUCTION_INTERFACE_MANTLE = 25

    SUBDUCTION_INTRASLAB_SUBDUCTION_INTERFACE = 32
    SUBDUCTION_INTRASLAB_MANTLE = 35

    SUBDUCTION_OUTERRISE_MANTLE = 45

    MANTLE_CRUSTAL = 51
    MANTLE_SUBDUCTION_INTERFACE = 52
    MANTLE_SUBDUCTION_INTRASLAB = 53
    MANTLE_SUBDUCTION_OUTERRISE = 54

    UNKNOWN = NO_VALUE

    @staticmethod
    def to_string(eq_type):
        return EqType.str_mapping().get(eq_type, "unknown")

    @staticmethod
    def str_mapping():
        return {
            EqType.UNKNOWN: "unknown",
            EqType.CRUSTAL: "crustal",
            EqType.SUBDUCTION_INTERFACE: "subduction_interface",
            EqType.SUBDUCTION_INTRASLAB: "subduction_intraslab",
            EqType.SUBDUCTION_OUTERRISE: "subduction_outerrise",
            EqType.MANTLE: "mantle",
            EqType.CRUSTAL_SUBDUCTION_INTERFACE: "crustal-subduction_interface",
            EqType.CRUSTAL_MANTLE: "crustal-mantle",
            EqType.SUBDUCTION_INTERFACE_CRUSTAL: "subduction_interface-crustal",
            EqType.SUBDUCTION_INTERFACE_SUBDUCTION_INTRASLAB: "subduction_interface-subduction_intraslab",
            EqType.SUBDUCTION_INTERFACE_MANTLE: "subduction_interface-mantle",
            EqType.SUBDUCTION_INTRASLAB_SUBDUCTION_INTERFACE: "subduction_intraslab-subduction_interface",
            EqType.SUBDUCTION_INTRASLAB_MANTLE: "subduction_intraslab-mantle",
            EqType.SUBDUCTION_OUTERRISE_MANTLE: "subduction_outerrise-mantle",
            EqType.MANTLE_CRUSTAL: "mantle-crustal",
            EqType.MANTLE_SUBDUCTION_INTERFACE: "mantle-subduction_interface",
            EqType.MANTLE_SUBDUCTION_INTRASLAB: "mantle-subduction_intraslab",
            EqType.MANTLE_SUBDUCTION_OUTERRISE: "mantle-subduction_outerrise",
        }

    @staticmethod
    def rename_category(config):
        """
        Re-name earthquake categories by what is specified in the config file's rename section.

        Function written with assistance from Kiro 2.16.2 claude-sonnet-5.
        """
        rename_config = config["rename"]
        name_to_eqtype = {
            name: eq_type for eq_type, name in EqType.str_mapping().items()
        }

        def _to_eqtype(value):
            name = value.split(".", 1)[1] if value.startswith("EqType.") else value
            return name_to_eqtype.get(
                name.lower(), EqType[name.upper().replace("-", "_")]
            )

        return {
            _to_eqtype(from_name): _to_eqtype(to_name)
            for from_name, to_name in rename_config.items()
        }
