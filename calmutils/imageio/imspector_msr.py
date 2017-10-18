from xml.etree import ElementTree
import struct


def _read_fmt(fd, fmt):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, fd.read(size))[0]


def _read_header(fd):

    FILE_MAGIC_STRING = b'OMAS_BF\n'
    MAGIC_NUMBER = 0xFFFF

    fmt_to_read = ['I', "Q"]

    magic_string_ = fd.read(len(FILE_MAGIC_STRING))
    magic_nr_ = _read_fmt(fd, 'H')

    if not ((magic_string_ == FILE_MAGIC_STRING) and (magic_nr_ == MAGIC_NUMBER)):
        return None

    version = _read_fmt(fd, 'I')
    first_stack_offset = _read_fmt(fd, 'Q')

    # skip 4 bytes
    fd.read(4);

    # read metadata offset
    metadata_offset = None
    if version > 1:
        metadata_offset = _read_fmt(fd, 'Q')

    # skip 5 bytes
    fd.read(5)

    old_version = _try_parse_root_header_desc(fd)

    if not old_version:
        fd.read(5)


    # desclen
    desc_len = _read_fmt(fd, "H")

    desc = fd.read(desc_len if old_version else (desc_len * 2))

    return (first_stack_offset, desc.decode('ascii' if old_version else 'utf_16_le'), metadata_offset)


def _try_parse_root_header_desc(fd):
    '''
    check if the file under investigation was written by imspector 0.11
    this function is to be called with fd at the point where 0.11 files will have a short desc_len
    followed by the ascii description (starting with <root)

    fd will remain at its original position
    '''
    init_pos = fd.tell()
    # skip desclen
    fd.seek(2, 1)
    # read desc start
    bts = fd.read(5)
    # jump back to where we were
    fd.seek(init_pos)
    # did we arrive at <root ?
    return bts == b'<root'


def _read_stack(fd, pos):
    STACK_MAGIC_STRING = b"OMAS_BF_STACK\n"
    MAGIC_NUMBER = 0xFFFF
    STACK_VERSION = 5
    MAXIMAL_NUMBER_OF_DIMENSIONS = 15

    fd.seek(pos)

    stack_magic_sting = fd.read(len(STACK_MAGIC_STRING))
    stack_magic_nr = _read_fmt(fd, 'H')
    stack_version = _read_fmt(fd, 'i')

    if not (stack_magic_sting == STACK_MAGIC_STRING and stack_magic_nr == MAGIC_NUMBER and stack_version <= STACK_VERSION):
        return None

    number_of_dimensions = _read_fmt(fd, "i")

    dims = []
    for i in range(MAXIMAL_NUMBER_OF_DIMENSIONS):
        d = _read_fmt(fd, "i")
        dims.append(d if i < number_of_dimensions else 1)

    lengths = []
    for i in range(MAXIMAL_NUMBER_OF_DIMENSIONS):
        d = _read_fmt(fd, "d")
        lengths.append(d if i < number_of_dimensions else 0.0)

    offsets = []
    for i in range(MAXIMAL_NUMBER_OF_DIMENSIONS):
        d = _read_fmt(fd, "d")
        offsets.append(d if i < number_of_dimensions else 0.0)


    pType = _read_fmt(fd, "i")
    compr = _read_fmt(fd, "i")

    fd.read(4)

    lengthOfName = _read_fmt(fd, "i")
    lengthOfDescription = _read_fmt(fd, "i")

    fd.read(8)

    lengthOfData = _read_fmt(fd, "q")
    nextPos =  _read_fmt(fd, "q")

    name = fd.read(lengthOfName)

    # TODO: this doesn't seem to do anything
    description = fd.read(lengthOfDescription)

    fd.seek(lengthOfData, 1)
    footerStart = fd.tell()

    footerSize = _read_fmt(fd, "I")
    fd.read(4*MAXIMAL_NUMBER_OF_DIMENSIONS)
    fd.read(4*MAXIMAL_NUMBER_OF_DIMENSIONS)
    firstMetaLen = _read_fmt(fd, "I")

    fd.seek(footerStart)
    fd.seek(footerSize, 1)

    for i in range(number_of_dimensions):
        nameLen = _read_fmt(fd, "I")
        fd.seek(nameLen,1)

    # skip first xml
    fd.seek(firstMetaLen, 1)


    # FIXME: hacky! seek until we find '<root>'
    accum = b''
    while not accum.endswith(b'<root'):
        accum += fd.read(1)

    fd.seek(-9, 1)

    sndMetaLen = _read_fmt(fd, "I")
    StackMeta = fd.read(sndMetaLen)

    return name, nextPos, StackMeta.decode('ascii')


def get_parameters_from_xml(xml, keys):
    '''
    find text values of xml elements
    :param xml: xml string
    :param keys: iterable of key strings
    :return: list of string values
    '''

    #TODO: check for XML errors
    res = []
    myET = ElementTree.fromstring(xml)

    for k in keys:
        res.append(str(myET.find(k).text))

    return res


def parse_msr(path):
    '''
    parse .msr file for metadata
    :param path: path to the file
    :return: header description, list (stack name, stack description (XML string))
    '''
    fd = open(path, "rb")
    next_offset, header_desc, meta_offset = _read_header(fd)
    # print hd
    res = []

    offset = next_offset
    while offset != 0:
        name, offset, StackMeta = _read_stack(fd, offset)
        res.append((name, StackMeta))

    fd.close()
    return header_desc, res