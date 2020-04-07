import os
import re
import pickle
import codecs
from sphinx_gallery import docs_resolv
from sphinx_gallery.docs_resolv import (
    SphinxDocLinkResolver,
    _handle_http_url_error,
    _sanitize_css_class,
)
from sphinx_gallery import sphinx_compatibility


def _embed_code_links(app, gallery_conf, gallery_dir):
    # Add resolvers for the packages for which we want to show links
    doc_resolvers = {}

    src_gallery_dir = os.path.join(app.builder.srcdir, gallery_dir)
    for this_module, url in gallery_conf["reference_url"].items():
        try:
            if url is None:
                doc_resolvers[this_module] = SphinxDocLinkResolver(
                    app.builder.outdir, src_gallery_dir, relative=True
                )
            else:
                doc_resolvers[this_module] = SphinxDocLinkResolver(url, src_gallery_dir)

        except (URLError, HTTPError) as e:
            _handle_http_url_error(e)

    html_gallery_dir = os.path.abspath(os.path.join(app.builder.outdir, gallery_dir))

    # patterns for replacement
    link_pattern = '<a href="{link}" title="{title}" class="{css_class}">{text}</a>'
    orig_pattern = '<span class="n">%s</span>'
    period = '<span class="o">.</span>'

    # This could be turned into a generator if necessary, but should be okay
    flat = [
        [dirpath, filename]
        for dirpath, _, filenames in os.walk(html_gallery_dir)
        for filename in filenames
    ]
    iterator = sphinx_compatibility.status_iterator(
        flat,
        "embedding documentation hyperlinks for %s... " % gallery_dir,
        color="fuchsia",
        length=len(flat),
        stringify_func=lambda x: os.path.basename(x[1]),
    )
    intersphinx_inv = getattr(app.env, "intersphinx_named_inventory", dict())
    builtin_modules = set(
        intersphinx_inv.get("python", dict()).get("py:module", dict()).keys()
    )
    for dirpath, fname in iterator:
        full_fname = os.path.join(html_gallery_dir, dirpath, fname)
        subpath = dirpath[len(html_gallery_dir) + 1 :]
        pickle_fname = os.path.join(
            src_gallery_dir, subpath, fname[:-5] + "_codeobj.pickle"
        )
        if not os.path.exists(pickle_fname):
            continue

        # we have a pickle file with the objects to embed links for
        with open(pickle_fname, "rb") as fid:
            example_code_obj = pickle.load(fid)
        # generate replacement strings with the links
        str_repl = {}
        for name in sorted(example_code_obj):
            cobjs = example_code_obj[name]
            # possible names from identify_names, which in turn gets
            # possibilites from NameFinder.get_mapping
            link = type_ = None
            for cobj in cobjs:
                for modname in (cobj["module_short"], cobj["module"]):
                    this_module = modname.split(".")[0]
                    cname = cobj["name"]

                    # Try doc resolvers first
                    if this_module in doc_resolvers:
                        try:
                            link, type_ = doc_resolvers[this_module].resolve(
                                cobj, full_fname, return_type=True
                            )
                        except (HTTPError, URLError) as e:
                            _handle_http_url_error(
                                e, msg="resolving %s.%s" % (modname, cname)
                            )

                    # next try intersphinx
                    if this_module == modname == "builtins":
                        this_module = "python"
                    elif modname in builtin_modules:
                        this_module = "python"
                    if link is None and this_module in intersphinx_inv:
                        inv = intersphinx_inv[this_module]
                        if modname == "builtins":
                            want = cname
                        else:
                            want = "%s.%s" % (modname, cname)
                        for key, value in inv.items():
                            # only python domain
                            if key.startswith("py") and want in value:
                                link = value[want][2]
                                type_ = key
                                break

                    # differentiate classes from instances
                    is_instance = (
                        type_ is not None
                        and "py:class" in type_
                        and not cobj["is_class"]
                    )

                    if link is not None:
                        # Add CSS classes
                        name_html = period.join(
                            orig_pattern % part for part in name.split(".")
                        )
                        full_function_name = "%s.%s" % (modname, cname)
                        css_class = "sphx-glr-backref-module-" + _sanitize_css_class(
                            modname
                        )
                        if type_ is not None:
                            css_class += (
                                " sphx-glr-backref-type-" + _sanitize_css_class(type_)
                            )
                        if is_instance:
                            css_class += " sphx-glr-backref-instance"
                        str_repl[name_html] = link_pattern.format(
                            link=link,
                            title=full_function_name,
                            css_class=css_class,
                            text=name_html,
                        )
                        break  # loop over possible module names

                if link is not None:
                    break  # loop over cobjs

        # do the replacement in the html file

        # ensure greediness
        names = sorted(str_repl, key=len, reverse=True)
        regex_str = "|".join(re.escape(name) for name in names)
        regex = re.compile(regex_str)

        def substitute_link(match):
            return str_repl[match.group()]

        if len(str_repl) > 0:
            with codecs.open(full_fname, "r", "utf-8") as fid:
                lines_in = fid.readlines()
            with codecs.open(full_fname, "w", "utf-8") as fid:
                for line in lines_in:
                    line_out = regex.sub(substitute_link, line)
                    fid.write(line_out)


docs_resolv._embed_code_links = _embed_code_links
