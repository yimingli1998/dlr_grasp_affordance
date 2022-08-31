import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
import io
import numpy as np
from .mjcf_utils import array_to_string, string_to_array, read_standard_xml
import transforms3d as T


class XMLError(Exception):
    """Exception raised for errors related to xml."""
    pass


class NameDuplicationError(Exception):
    """Exception raised for duplicated names."""
    pass


class ModelConflictError(Exception):
    """Exception raised for a model with conflicted definition"""


class MujocoXML(object):
    """
    Base class of Mujoco xml file
    Wraps around ElementTree and provides additional functionality for merging different models.
    Specially, we keep track of <worldbody/>, <actuator/> and <asset/>
    """

    def __init__(self, fname, normalize_names=True):
        """
        Loads a mujoco xml from file.
        Args:
            fname (str): path to the MJCF xml file.
        """
        self.file = fname
        self.folder = os.path.dirname(fname)
        self.root = read_standard_xml(fname)
        self.name = self.root.get("model")
        self.worldbody = self.create_default_element("worldbody")
        self.actuator = self.create_default_element("actuator")
        self.sensor = self.create_default_element("sensor")
        self.asset = self.create_default_element("asset")
        self.tendon = self.create_default_element("tendon")
        self.equality = self.create_default_element("equality")
        self.contact = self.create_default_element("contact")
        self.default = self.create_default_element("default")
        self.compiler = self.create_default_element('compiler')
        self.option = self.create_default_option()
        self.resolve_asset_dependency()
        if normalize_names:
            self.normalize_names()

    def resolve_asset_dependency(self):
        """
        Converts every file dependency into absolute path so when we merge we don't break things.
        """

        for node in self.asset.findall("./*[@file]"):
            file = node.get("file")
            abs_path = os.path.abspath(self.folder)
            abs_path = os.path.join(abs_path, file)
            node.set("file", abs_path)

    def create_default_element(self, name):
        """
        Creates a <@name/> tag under root if there is none.
        """

        found = self.root.find(name)
        if found is not None:
            return found
        ele = ET.Element(name)
        self.root.append(ele)
        return ele

    def create_default_option(self):
        elem = self.create_default_element('option')
        elem.set('integrator', 'RK4')
        return elem

    def rename(self, name, normalize=True):
        self.name = name
        if normalize:
            self.normalize_names()

    def merge(self, others, merge_body=True):
        """
        Default merge method.
        Args:
            others: another MujocoXML instance
                raises XML error if @other is not a MujocoXML instance.
                merges <worldbody/>, <actuator/> and <asset/> of @other into @self
            merge_body: True if merging child bodies of @other. Defaults to True.
        """
        if type(others) is not list:
            others = [others]
        for idx, other in enumerate(others):
            if not isinstance(other, MujocoXML):
                raise XMLError("{} is not a MujocoXML instance.".format(type(other)))
            if merge_body:
                for body in other.worldbody:
                    self.worldbody.append(body)
            self.merge_asset(other)
            for one_actuator in other.actuator:
                self.actuator.append(one_actuator)
            for one_sensor in other.sensor:
                self.sensor.append(one_sensor)
            for one_tendon in other.tendon:
                self.tendon.append(one_tendon)
            for one_equality in other.equality:
                self.equality.append(one_equality)
            for one_contact in other.contact:
                self.contact.append(one_contact)
            for one_default in other.default:
                self.default.append(one_default)
            for one_compiler in other.compiler:
                self.default.append(one_compiler)
        # self.configs.append(other.configs)

    def get_model(self, mode="mujoco_py"):
        """
        Returns a MjModel instance from the current xml tree.
        """

        available_modes = ["mujoco_py"]
        with io.StringIO() as string:
            string.write(ET.tostring(self.root, encoding="unicode"))
            if mode == "mujoco_py":
                from mujoco_py import load_model_from_xml

                model = load_model_from_xml(string.getvalue())
                return model
            raise ValueError(
                "Unkown model mode: {}. Available options are: {}".format(
                    mode, ",".join(available_modes)))

    def get_xml(self):
        """
        Returns a string of the MJCF XML file.
        """
        with io.StringIO() as string:
            string.write(ET.tostring(self.root, encoding="unicode"))
            return string.getvalue()

    def save_model(self, fname, pretty=False):
        """
        Saves the xml to file.
        Args:
            fname: output file location
            pretty: attempts!! to pretty print the output
        """
        with open(fname, "w") as f:
            xml_str = ET.tostring(self.root, encoding="unicode")
            if pretty:
                parsed_xml = xml.dom.minidom.parseString(xml_str)
                xml_str = parsed_xml.toprettyxml(newl="")
            f.write(xml_str)

    def merge_asset(self, other):
        """
        Useful for merging other files in a custom logic.
        """
        for asset in other.asset:
            asset_name = asset.get("name")
            asset_type = asset.tag
            # Avoids duplication
            pattern = "./{}[@name='{}']".format(asset_type, asset_name)
            if self.asset.find(pattern) is None:
                self.asset.append(asset)

    def get_element_names(self, root, element_type):
        """
        Searches recursively through the @root and returns a list of names of the specified @element_type
        Args:
            root (ET.Element): Root of the xml element tree to start recursively searching through
                (e.g.: `self.worldbody`)
            element_type (str): Name of element to return names of. (e.g.: "site", "geom", etc.)
        Returns:
            list: names that correspond to the specified @element_type
        """
        names = []
        for child in root:
            if child.tag == element_type:
                names.append(child.get("name"))
            names += self.get_element_names(child, element_type)
        return names

    def add_prefix(self,
                   prefix,
                   tags=("body", "joint", "sensor", "site", "geom", "camera", "actuator", "tendon", "asset", "mesh",
                         "texture", "material")):
        """
        Utility method to add prefix to all body names to prevent name clashes
        Args:
            prefix (str): Prefix to be appended to all requested elements in this XML
            tags (list or tuple): Tags to be searched in the XML. All elements with specified tags will have "prefix"
                prepended to it
        """
        # Define tags as a set
        tags = set(tags)

        # Define equalities set to pass at the end
        equalities = set(tags)

        # Add joints to equalities if necessary
        if "joint" in tags:
            equalities = equalities.union(["joint1", "joint2"])

        # Handle actuator elements
        if "actuator" in tags:
            tags.discard("actuator")
            for actuator in self.actuator:
                self._add_prefix_recursively(actuator, tags, prefix)

        # Handle sensor elements
        if "sensor" in tags:
            tags.discard("sensor")
            for sensor in self.sensor:
                self._add_prefix_recursively(sensor, tags, prefix)

        # Handle tendon elements
        if "tendon" in tags:
            tags.discard("tendon")
            for tendon in self.tendon:
                self._add_prefix_recursively(tendon, tags.union(["fixed"]), prefix)
            # Also take care of any tendons in equality constraints
            equalities = equalities.union(["tendon1", "tendon2"])

        # Handle asset elements
        if "asset" in tags:
            tags.discard("asset")
            for asset in self.asset:
                if asset.tag in tags:
                    self._add_prefix_recursively(asset, tags, prefix)

        # Handle contacts and equality names for body elements
        if "body" in tags:
            for contact in self.contact:
                if "body1" in contact.attrib:
                    contact.set("body1", prefix + contact.attrib["body1"])
                if "body2" in contact.attrib:
                    contact.set("body2", prefix + contact.attrib["body2"])
            # Also take care of any bodies in equality constraints
            equalities = equalities.union(["body1", "body2"])

        # Handle all equality elements
        for equality in self.equality:
            self._add_prefix_recursively(equality, equalities, prefix)

        # Handle all remaining bodies in the element tree
        for body in self.worldbody:
            if body.tag in tags:
                self._add_prefix_recursively(body, tags, prefix)

    def _add_prefix_recursively(self, root, tags, prefix):
        """
        Iteratively searches through all children nodes in "root" element to append "prefix" to any named subelements
        with a tag in "tags"
        Args:
            root (ET.Element): Root of the xml element tree to start recursively searching through
                (e.g.: `self.worldbody`)
            tags (list or tuple): Tags to be searched in the XML. All elements with specified tags will have "prefix"
                prepended to it
            prefix (str): Prefix to be appended to all requested elements in this XML
        """
        # First re-name this element
        if "name" in root.attrib:
            root.set("name", prefix + root.attrib["name"])

        # Then loop through all tags and rename any appropriately
        for tag in tags:
            if tag in root.attrib:
                root.set(tag, prefix + root.attrib[tag])

        # Recursively go through child elements
        for child in root:
            if child.tag in tags:
                self._add_prefix_recursively(child, tags, prefix)

    def normalize_names(self):
        """
        Add model name to all names as prefix.
        """
        for node in self.asset.findall(".//*[@name]"):
            name = node.get("name")
            if not name.startswith(self.name + "."):
                node.set("name", self.name + "." + name)

        for attr in ['texture', 'material', 'mesh']:
            for node in self.root.findall(".//*[@{}]".format(attr)):
                name = node.get(attr)
                if not name.startswith(self.name + "."):
                    node.set(attr, self.name + "." + name)

        for node in self.worldbody.findall(".//*[@name]"):
            name = node.get("name")
            if not name.startswith(self.name + "."):
                node.set("name", self.name + "." + name)

        for node in self.worldbody.findall(".//*[@joint]"):
            joint = node.get("joint")
            if not joint.startswith(self.name + "."):
                node.set("joint", self.name + "." + name)

    def check_name_duplication(self, other):
        """
        Check if name duplication occurs and raise error.
        """
        self_names = set(
            [node.get("name") for node in self.root.findall("./*[@name]")])
        other_names = set(
            [node.get("name") for node in other.root.findall("./*[@name]")])
        if len(set.intersection(self_names, other_names)):
            raise NameDuplicationError()

    def translate(self, offset):
        """
        Move the entire scene by offset
        """
        for body in self.worldbody:
            pos = body.get("pos", "0 0 0")
            pos = string_to_array(pos)
            pos += offset
            body.set("pos", array_to_string(pos))

    def rotate(self, euler_xyz_degree):
        """
        Rotate the entire scene by euler angles
        """
        degree = True
        compiler = self.root.find('.//compiler[@angle]')
        if compiler and compiler.get('angle'):
            if compiler.get('angle') == 'radian':
                degree = False

        x, y, z = np.array(euler_xyz_degree) * np.pi / 180
        R = T.euler.euler2quat(x, y, z, 'sxyz')

        if self.root.find('.//compiler[@eulerseq]'):
            raise NotImplementedError()

        for body in self.worldbody:
            if body.tag == 'light':
                continue
            quat = None
            if body.get('axisangle'):
                axisangle = string_to_array(body.get('axisangle'))
                length = np.linalg.norm(axisangle)
                quat = T.quaternions.axangle2quat(axisangle / length, length)
                body.set('axisangle', None)
            elif body.get('xyaxes'):
                raise NotImplementedError()
            elif body.get('zaxis'):
                raise NotImplementedError()
            elif body.get('euler'):
                i, j, k = string_to_array(body.get('euler'))
                if degree:
                    i *= np.pi / 180
                    j *= np.pi / 180
                    k *= np.pi / 180
                quat = T.euler.euler2quat(i, j, k, 'sxyz')
                body.set('euler', None)
            else:
                quat = string_to_array(body.get('quat', '1 0 0 0'))
                body.set('quat', None)

            quat = T.quaternions.qmult(R, quat)
            body.set('quat', array_to_string(quat))
