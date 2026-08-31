import 'dart:math' as math;

/// Dart port of ml_pipeline/preprocessing/features_v3.py.
///
/// MUST stay byte-equivalent to the Python original. The model is trained on
/// whatever that file produces, so any divergence here puts inference in a
/// different feature space than training -- the failure that once measured
/// 99.0% to 13.5% on the word model. Constants are duplicated from the Python
/// deliberately rather than derived, so a change there is a visible change here.
///
/// Input is the 226-d canonicalised, wrist-normalised vector the existing
/// pipeline already produces. Output is 736-d:
///
///   body-normalised pose xyz          75    location
///   hand blocks / primary hand span  126    inter-hand geometry
///   hand-local shapes                126    handshape
///   distances                          5    location contrasts
///   hand-detected flags                2    presence
///   velocity  (dx  over 1 frame)     201    movement
///   accel     (dx2 over 2 frames)    201    movement, wider baseline
///   arm + palm orientation            18    limb slant, elbow flexion
class FeaturesV3 {
  static const int dim = 754;

  static const int _pose0 = 126;
  static const int _nose = 0;
  static const int _lShoulder = 11;
  static const int _rShoulder = 12;
  static const int _lWrist = 15;
  static const int _rWrist = 16;
  static const int _lElbow = 13;
  static const int _rElbow = 14;
  // Hand landmarks: wrist, index knuckle, middle knuckle, pinky knuckle.
  static const int _hWrist = 0;
  static const int _hIndexMcp = 5;
  static const int _hMiddleMcp = 9;
  static const int _hPinkyMcp = 17;

  static const double _minWidth = 0.04;
  static const double _eps = 1e-6;

  /// MediaPipe normalises x to image width and y to image height separately, so
  /// on a 4:3 canvas an x-unit covers 4/3 as many pixels as a y-unit. Rescaling
  /// x makes every distance, span and rotation below isotropic.
  static const double _aspect = 640.0 / 480.0;

  /// numpy's rint: round half to EVEN, not half away from zero.
  ///
  /// Dart's .round() rounds 0.5 up while numpy rounds it to the nearest even
  /// integer, so any resample index landing exactly on a half picks a different
  /// frame in the two implementations. On a 30-frame sequence that happens
  /// often enough to shift whole frames.
  static int _rint(double x) {
    final f = x.floorToDouble();
    final diff = x - f;
    final fi = f.toInt();
    if (diff > 0.5) return fi + 1;
    if (diff < 0.5) return fi;
    return fi.isEven ? fi : fi + 1;
  }

  /// Motion of the hand blocks between consecutive frames.
  static List<double> motionEnergy(List<List<double>> frames) {
    final n = frames.length;
    final e = List<double>.filled(n, 0.0);
    for (int t = 1; t < n; t++) {
      double s = 0.0;
      for (int i = 0; i < 126; i++) {
        final d = frames[t][i] - frames[t - 1][i];
        s += d * d;
      }
      e[t] = math.sqrt(s);
    }
    if (n > 1) e[0] = e[1];
    return e;
  }

  /// Drop leading and trailing frames below [frac] of peak motion, then
  /// resample what remains back to the original length.
  ///
  /// This is motion-energy segmentation applied inside the captured window.
  /// The camera catches the signer settling before a sign and holding after it,
  /// and those frames spend sequence budget on stillness while teaching the
  /// model that a sign begins with a static hold.
  static List<List<double>> trimIdle(List<List<double>> frames,
      {double frac = 0.15, int minKeep = 8}) {
    final n = frames.length;
    if (n < minKeep) return frames;
    final e = motionEnergy(frames);
    double peak = 0.0;
    for (final v in e) {
      if (v > peak) peak = v;
    }
    if (peak <= _eps) return frames;

    // Count of frames above threshold, NOT the span between the first and
    // last. The Python guard is len(live) and the two differ whenever the live
    // frames are non-contiguous, which is most clips.
    int a = -1, b = -1, live = 0;
    for (int t = 0; t < n; t++) {
      if (e[t] >= frac * peak) {
        if (a < 0) a = t;
        b = t;
        live++;
      }
    }
    if (a < 0 || live < minKeep) return frames;

    final segLen = b - a + 1;
    return List<List<double>>.generate(n, (i) {
      final idx = a + _rint((segLen - 1) * i / (n - 1));
      return frames[idx];
    });
  }

  /// frames: [T][226] to [T][736]
  static List<List<double>> build(List<List<double>> frames) {
    final t = frames.length;

    // Positions that velocity and acceleration are differenced from: the
    // body-normalised pose followed by hand_loc, matching the Python
    // concatenate order exactly.
    final pos =
        List<List<double>>.generate(t, (_) => List<double>.filled(201, 0.0));
    final out =
        List<List<double>>.generate(t, (_) => List<double>.filled(dim, 0.0));

    for (int f = 0; f < t; f++) {
      final src = frames[f];
      final v = List<double>.filled(226, 0.0);
      final lim = src.length < 226 ? src.length : 226;
      for (int i = 0; i < lim; i++) {
        v[i] = src[i];
      }

      // Square-pixel coordinates, before anything geometric is computed.
      for (int i = 0; i < 126; i += 3) {
        v[i] *= _aspect;
      }
      for (int k = 0; k < 25; k++) {
        v[_pose0 + 4 * k] *= _aspect;
      }

      double px(int j) => v[_pose0 + 4 * j];
      double py(int j) => v[_pose0 + 4 * j + 1];
      double pz(int j) => v[_pose0 + 4 * j + 2];

      final lsx = px(_lShoulder), lsy = py(_lShoulder);
      final rsx = px(_rShoulder), rsy = py(_rShoulder);
      final cx = (lsx + rsx) / 2.0, cy = (lsy + rsy) / 2.0;
      final w =
          math.sqrt((lsx - rsx) * (lsx - rsx) + (lsy - rsy) * (lsy - rsy));
      final ok = w > _minWidth;
      final wsafe = ok ? w : 1.0;

      final o = out[f];
      final p = pos[f];

      // body-normalised pose, 0..75
      final bxs = List<double>.filled(25, 0.0);
      final bys = List<double>.filled(25, 0.0);
      for (int k = 0; k < 25; k++) {
        final bx = ok ? (px(k) - cx) / wsafe : 0.0;
        final by = ok ? (py(k) - cy) / wsafe : 0.0;
        final bz = ok ? pz(k) / wsafe : 0.0;
        bxs[k] = bx;
        bys[k] = by;
        o[k * 3] = bx;
        o[k * 3 + 1] = by;
        o[k * 3 + 2] = bz;
        p[k * 3] = bx;
        p[k * 3 + 1] = by;
        p[k * 3 + 2] = bz;
      }

      // hand spans and presence
      final present = <bool>[false, false];
      final span = <double>[0.0, 0.0];
      for (int h = 0; h < 2; h++) {
        final base = h * 63;
        bool any = false;
        for (int i = 0; i < 63; i++) {
          if (v[base + i] != 0.0) {
            any = true;
            break;
          }
        }
        present[h] = any;
        if (!any) continue;
        final wx = v[base], wy = v[base + 1];
        double mx = 0.0;
        for (int j = 0; j < 21; j++) {
          final dx = v[base + j * 3] - wx;
          final dy = v[base + j * 3 + 1] - wy;
          final d = math.sqrt(dx * dx + dy * dy);
          if (d > mx) mx = d;
        }
        span[h] = mx;
      }

      // Hands are scaled by a HAND quantity, never by shoulder width. The
      // device restores hand scale through the second landmarker pass while the
      // letterbox leaves the body compressed, so any hand-over-body ratio is
      // 1.78x wrong on device while looking healthy on the corpus. Measured
      // twice: 65.36% to 30.77%, then 73.59% to 33.24%.
      final ref = span[1] > _eps ? span[1] : (span[0] > _eps ? span[0] : 1.0);

      // hand_loc 75..201, hand_shape 201..327
      for (int h = 0; h < 2; h++) {
        if (!present[h]) continue;
        final base = h * 63;
        final locBase = 75 + h * 63;
        final shapeBase = 201 + h * 63;
        final wx = v[base], wy = v[base + 1], wz = v[base + 2];
        final s = span[h] > _eps ? span[h] : 1.0;
        for (int i = 0; i < 63; i++) {
          final loc = v[base + i] / ref;
          o[locBase + i] = loc;
          p[75 + h * 63 + i] = loc;
        }
        for (int j = 0; j < 21; j++) {
          o[shapeBase + j * 3] = (v[base + j * 3] - wx) / s;
          o[shapeBase + j * 3 + 1] = (v[base + j * 3 + 1] - wy) / s;
          o[shapeBase + j * 3 + 2] = (v[base + j * 3 + 2] - wz) / s;
        }
      }

      // distances 327..332
      if (ok) {
        double bxi(int j) => (px(j) - cx) / wsafe;
        double byi(int j) => (py(j) - cy) / wsafe;
        double hyp(double a, double b) => math.sqrt(a * a + b * b);
        final nx = bxi(_nose), ny = byi(_nose);
        final lwx = bxi(_lWrist), lwy = byi(_lWrist);
        final rwx = bxi(_rWrist), rwy = byi(_rWrist);
        o[327] = hyp(lwx - nx, lwy - ny);
        o[328] = hyp(rwx - nx, rwy - ny);
        o[329] = hyp(lwx, lwy);
        o[330] = hyp(rwx, rwy);
        o[331] = hyp(lwx - rwx, lwy - rwy);
      }

      // flags 332..334
      o[332] = present[0] ? 1.0 : 0.0;
      o[333] = present[1] ? 1.0 : 0.0;

      // arm and palm orientation 736..754
      //
      // Arm geometry is implicit in body_pose, but the model would have to
      // infer an ANGLE from raw coordinates. Signs sharing a trajectory and
      // differing only in forearm slant are exactly what gets confused --
      // HELLO and FUTURE travel the same path and differ in whether the
      // forearm is angled or straight. Unit vectors, not angles, so there is
      // no wraparound and a zero-length limb degrades to zeros.
      int ai = 736;
      const arms = [
        [_lShoulder, _lElbow, _lWrist],
        [_rShoulder, _rElbow, _rWrist],
      ];
      for (final a in arms) {
        final sh = a[0], el = a[1], wr = a[2];
        var ux = bxs[el] - bxs[sh], uy = bys[el] - bys[sh];
        final un = math.sqrt(ux * ux + uy * uy);
        if (un > _eps) {
          ux /= un;
          uy /= un;
        } else {
          ux = 0.0;
          uy = 0.0;
        }
        var fx = bxs[wr] - bxs[el], fy = bys[wr] - bys[el];
        final fn = math.sqrt(fx * fx + fy * fy);
        if (fn > _eps) {
          fx /= fn;
          fy /= fn;
        } else {
          fx = 0.0;
          fy = 0.0;
        }
        o[ai++] = ux;
        o[ai++] = uy;
        o[ai++] = fx;
        o[ai++] = fy;
        o[ai++] = ux * fx + uy * fy;      // elbow flexion: +1 straight
        o[ai++] = ok ? fn : 0.0;          // forearm length in shoulder widths
      }
      for (int h = 0; h < 2; h++) {
        final sb = 201 + h * 63;
        var ax = o[sb + _hMiddleMcp * 3] - o[sb + _hWrist * 3];
        var ay = o[sb + _hMiddleMcp * 3 + 1] - o[sb + _hWrist * 3 + 1];
        final an = math.sqrt(ax * ax + ay * ay);
        if (an > _eps) {
          ax /= an;
          ay /= an;
        } else {
          ax = 0.0;
          ay = 0.0;
        }
        final sx = o[sb + _hIndexMcp * 3] - o[sb + _hPinkyMcp * 3];
        final sy = o[sb + _hIndexMcp * 3 + 1] - o[sb + _hPinkyMcp * 3 + 1];
        o[ai++] = ax;
        o[ai++] = ay;
        // Knuckle spread separates an open hand from a fist at the same angle.
        o[ai++] = present[h] ? math.sqrt(sx * sx + sy * sy) : 0.0;
      }
    }

    // velocity 334..535 (one-frame difference)
    for (int f = 1; f < t; f++) {
      for (int i = 0; i < 201; i++) {
        out[f][334 + i] = pos[f][i] - pos[f - 1][i];
      }
    }
    // acceleration 535..736 (two-frame difference)
    for (int f = 2; f < t; f++) {
      for (int i = 0; i < 201; i++) {
        out[f][535 + i] = pos[f][i] - pos[f - 2][i];
      }
    }

    for (final row in out) {
      for (int i = 0; i < dim; i++) {
        final x = row[i];
        if (x.isNaN || x.isInfinite) row[i] = 0.0;
      }
    }
    return out;
  }
}
