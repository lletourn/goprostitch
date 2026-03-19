import audalign as ad
import json
import sys
v1 = sys.argv[1]
v2 = sys.argv[2]

fingerprint_rec = ad.FingerprintRecognizer()
correlation_rec = ad.CorrelationRecognizer()
cor_spec_rec = ad.CorrelationSpectrogramRecognizer()
visual_rec = ad.VisualRecognizer()

fingerprint_rec.config.set_accuracy(3)

results = ad.align_files(
    v1,
    v2,
    recognizer=correlation_rec
)

fine_results = ad.fine_align(results, recognizer=cor_spec_rec)

print(fine_results)
