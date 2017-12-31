import xml.etree.ElementTree as ET

def xml2dict(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    def parse_node(node):
        tree = {}
        for child in node:
            _tag  = child.tag
            _text = child.text.strip()
            _node = parse_node(child)
            
            if _node:
                v = _node
            else:
                v = _text

            if _tag in tree.keys():
                if type(tree[_tag]) is list:
                    tree[_tag].append(v)
                else:
                    tree[_tag] = [tree[_tag], v]
            else:
                tree[_tag] = v
        return tree

    return parse_node(root)

