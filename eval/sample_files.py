from common import *
c = conn(); cur = c.cursor()
cur.execute("select setseed(0.42)")
cur.execute("""select file_path, doc_type, coalesce(domain,'-'), chunk_count from source_files
 where coalesce(domain,'') <> 'timebound_context' and chunk_count>=2 and doc_type='markdown' order by random() limit 40""")
for r in cur.fetchall():
    print('md', r[2][:18].ljust(18), r[3], r[0][38:])
cur.execute("select file_path, chunk_count from source_files where doc_type='pdf' order by random() limit 65")
for r in cur.fetchall():
    print('pdf', r[1], Path(r[0]).name)
