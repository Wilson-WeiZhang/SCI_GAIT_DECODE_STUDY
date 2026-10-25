function write_align_log_json(align_log, out_json_path)
%WRITE_ALIGN_LOG_JSON  Serialize align_log struct to JSON (pretty-printed).
    txt = jsonencode(align_log, 'PrettyPrint', true);
    fid = fopen(out_json_path, 'w');
    assert(fid > 0, 'Could not open %s for writing', out_json_path);
    fprintf(fid, '%s', txt);
    fclose(fid);
end
