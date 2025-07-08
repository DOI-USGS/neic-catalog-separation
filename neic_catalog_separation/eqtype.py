from enum import Enum

NO_VALUE = 999


class EqType(Enum):
    """Enumerated values for types of earthquakes."""

    CRUSTAL = 0
    SUBDUCTION_INTERFACE = 10
    SUBDUCTION_INTRASLAB = 11
    SUBDUCTION_OUTERRISE = 12
    SUBDUCTION_INTERFACE_MANTLE = 20
    SUBDUCTION_INTRASLAB_MANTLE = 21
    SUBDUCTION_OUTERRISE_MANTLE = 22
    UNKNOWN = NO_VALUE

    @staticmethod
    def flag_mantle(eq_type):
        mantle = EqType.mantle_mapping()
        if mantle is not None:
            return mantle.get(eq_type, EqType.UNKNOWN)
        else:
            return EqType.UNKNOWN

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
            EqType.SUBDUCTION_INTERFACE_MANTLE: "subduction_interface_mantle",
            EqType.SUBDUCTION_INTRASLAB_MANTLE: "subduction_intraslab_mantle",
            EqType.SUBDUCTION_OUTERRISE_MANTLE: "subduction_outerrise_mantle",
        }

    @staticmethod
    def mantle_mapping():
        TO_MANTLE = {
            EqType.SUBDUCTION_INTERFACE: EqType.SUBDUCTION_INTERFACE_MANTLE,
            EqType.SUBDUCTION_INTRASLAB: EqType.SUBDUCTION_INTRASLAB_MANTLE,
            EqType.SUBDUCTION_OUTERRISE: EqType.SUBDUCTION_OUTERRISE_MANTLE,
        }
