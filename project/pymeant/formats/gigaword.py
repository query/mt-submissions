"""Tools for working with the Gigaword corpus."""


import re
try:
    from xml.etree import cElementTree as ElementTree
except ImportError:
    from xml.etree import ElementTree


def paragraphs(sgml_file):
    """Yield paragraphs from the Gigaword SGML file *f*."""
    sgml = sgml_file.read()
    # Massage the SGML markup so it resembles valid XML.
    #
    # Gigaword isn't consistent about its use of entities, and when
    # it does use them they're capitalized.  Bare ampersands and the
    # "&AMP;" capitalization aren't valid in XML, so we convert 'em.
    sgml = re.sub(r'&amp;', '&', sgml, re.IGNORECASE)
    sgml = re.sub(r'&', '&amp;', sgml)
    # Add an XML encoding declaration so the parser doesn't choke when
    # it encounters a non-ASCII character, and wrap the document with a
    # fake root element so the parser doesn't balk at its absence.
    sgml = ('<?xml version="1.0" encoding="iso-8859-1"?>' +
            '<ROOT>' + sgml + '</ROOT>')
    document = ElementTree.XML(sgml)
    for p in document.iter('P'):
        yield p.text


def main():
    """Command line entry point."""
    import gzip
    import sys

    if len(sys.argv) < 2:
        print >> sys.stderr, ('Usage: {} GIGAWORD_SGML_GZ [...]'
                              .format(sys.argv[0]))
        return

    for filename in sys.argv[1:]:
        try:
            with gzip.open(filename) as f:
                for paragraph in paragraphs(f):
                    print paragraph.encode('utf-8')
        except IOError as e:
            print >> sys.stderr, 'Error reading {}: {!s}'.format(filename, e)


if __name__ == '__main__':
    main()
