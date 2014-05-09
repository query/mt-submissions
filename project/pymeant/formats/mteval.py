"""Tools for working with MTEval XML files, such as those in the DARPA
GALE corpus."""


try:
    from xml.etree import cElementTree as ElementTree
except ImportError:
    from xml.etree import ElementTree


def segments(xml_file):
    """Yield the text of segments from the MTEval XML file *f*."""
    document = ElementTree.parse(xml_file)
    for segment in document.iter('seg'):
        yield segment.text


def main():
    """Command line entry point."""
    import sys

    if len(sys.argv) < 2:
        print >> sys.stderr, ('Usage: {} MTEVAL_XML [...]'
                              .format(sys.argv[0]))
        return

    for filename in sys.argv[1:]:
        try:
            with open(filename) as f:
                for segment in segments(f):
                    print segment.encode('utf-8')
        except IOError as e:
            print >> sys.stderr, 'Error reading {}: {!s}'.format(filename, e)


if __name__ == '__main__':
    main()
