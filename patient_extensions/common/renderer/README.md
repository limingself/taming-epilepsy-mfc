# Public O/F/R/C renderer

`render_channel_plate.py` and `config.json` are the common renderer and frozen
figure contract used for HUP065 and HUP080. The renderer expects a locally
generated, independently bound candidate payload. That signal payload and its
one-time binding receipt are not distributed here.

The committed PDF/PNG files under each patient directory are the accepted
publication exports. `verify_public_results.py` checks their hashes against the
public QA records without loading patient signal arrays.

