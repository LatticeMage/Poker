// Host-side range input for the nvcc tools: a small JSON reader and the hand-list grammar of
// src/lib/poker/spots.ts (parseWeighted), so the tools read src/lib/poker/data/charts.json and spots.json
// directly instead of keeping a second copy of the ranges.
//
// A range is 169 numbers in ALL_HANDS order (poker.cuh class order): the share of that class's combos in the
// list, 0..1 ("AsAc" alone is 1 of 6 combos of AA). Needs poker.cuh included first (class names).
#pragma once

#include <map>
#include <regex>
#include <stdexcept>

// -- JSON ---------------------------------------------------------------------------------------
struct Json {
	enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
	double num = 0;
	std::string str;
	std::vector<Json> arr;
	std::vector<std::pair<std::string, Json>> obj;

	const Json& at(const std::string& key) const {
		for (const auto& kv : obj)
			if (kv.first == key) return kv.second;
		throw std::runtime_error("missing JSON key " + key);
	}
	bool has(const std::string& key) const {
		for (const auto& kv : obj)
			if (kv.first == key) return true;
		return false;
	}
};

struct JsonReader {
	const std::string& s;
	size_t i = 0;
	explicit JsonReader(const std::string& text) : s(text) {}
	void ws() { while (i < s.size() && isspace((unsigned char)s[i])) i++; }
	[[noreturn]] void fail(const char* what) { throw std::runtime_error(std::string("JSON: ") + what + " at " + std::to_string(i)); }
	std::string string_() {
		std::string out;
		i++;  // opening quote
		while (i < s.size() && s[i] != '"') {
			if (s[i] == '\\' && i + 1 < s.size()) {
				i++;
				switch (s[i]) {
					case 'n': out += '\n'; break;
					case 't': out += '\t'; break;
					case 'u': out += '?'; i += 4; break;  // not needed by our files
					default: out += s[i];
				}
			} else {
				out += s[i];
			}
			i++;
		}
		if (i >= s.size()) fail("unterminated string");
		i++;
		return out;
	}
	Json value() {
		ws();
		if (i >= s.size()) fail("unexpected end");
		Json j;
		char c = s[i];
		if (c == '{') {
			j.type = Json::Obj;
			i++;
			ws();
			if (s[i] == '}') { i++; return j; }
			for (;;) {
				ws();
				if (s[i] != '"') fail("object key");
				std::string key = string_();
				ws();
				if (s[i++] != ':') fail("colon");
				j.obj.emplace_back(key, value());
				ws();
				if (s[i] == ',') { i++; continue; }
				if (s[i] == '}') { i++; break; }
				fail("object");
			}
		} else if (c == '[') {
			j.type = Json::Arr;
			i++;
			ws();
			if (s[i] == ']') { i++; return j; }
			for (;;) {
				j.arr.push_back(value());
				ws();
				if (s[i] == ',') { i++; continue; }
				if (s[i] == ']') { i++; break; }
				fail("array");
			}
		} else if (c == '"') {
			j.type = Json::Str;
			j.str = string_();
		} else if (!s.compare(i, 4, "true")) {
			j.type = Json::Bool; j.num = 1; i += 4;
		} else if (!s.compare(i, 5, "false")) {
			j.type = Json::Bool; i += 5;
		} else if (!s.compare(i, 4, "null")) {
			i += 4;
		} else {
			char* end;
			j.type = Json::Num;
			j.num = strtod(s.c_str() + i, &end);
			if (end == s.c_str() + i) fail("number");
			i = end - s.c_str();
		}
		return j;
	}
};

static std::string read_file(const std::string& path) {
	FILE* f = fopen(path.c_str(), "rb");
	if (!f) throw std::runtime_error("cannot read " + path);
	std::string out;
	char buf[65536];
	size_t n;
	while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
	fclose(f);
	return out;
}

static Json json_load(const std::string& path) {
	std::string text = read_file(path);
	JsonReader r(text);
	return r.value();
}

// -- hand lists ---------------------------------------------------------------------------------
static int rank_of(char c) {
	const char* p = strchr(RANKS, c);
	return p && c ? (int)(p - RANKS) : -1;  // 0 = deuce .. 12 = ace
}

static int combos_of_class(int cls) {
	const std::string& n = g_names[cls];
	return n.size() == 2 ? 6 : (n[2] == 's' ? 4 : 12);
}

static int class_of(int r1, int r2, bool suited) {
	int hi = std::max(r1, r2), lo = std::min(r1, r2);
	std::string name = hi == lo ? std::string(2, RANKS[hi]) : std::string() + RANKS[hi] + RANKS[lo] + (suited ? 's' : 'o');
	return class_index(name);
}

// Same grammar as parseWeighted in spots.ts: "QQ-88", "TT+", "AQs+", "A5s-A2s", "JTs-98s", "AK", "KJo+",
// single combos such as "AsAc". Returns the share of each class's combos named by the list.
static std::vector<double> parse_weighted(const std::string& text) {
	static const std::string R = "[2-9TJQKA]";
	static const std::regex COMBO("^(" + R + ")([shdc])(" + R + ")([shdc])$");
	static const std::regex PAIR("^(" + R + ")\\1(\\+)?$");
	static const std::regex PAIR_SPAN("^(" + R + ")\\1-(" + R + ")\\2$");
	static const std::regex HAND("^(" + R + ")(" + R + ")([so]?)(\\+)?$");
	static const std::regex SPAN("^(" + R + ")(" + R + ")([so]?)-(" + R + ")(" + R + ")([so]?)$");
	std::vector<double> count(NCLASS, 0.0);
	auto add = [&](int cls, double n) { count[cls] += n; };
	auto suffixes = [](const std::string& flag) {
		return flag.empty() ? std::vector<bool>{true, false} : std::vector<bool>{flag == "s"};
	};
	size_t pos = 0;
	while (pos <= text.size()) {
		size_t comma = text.find(',', pos);
		if (comma == std::string::npos) comma = text.size();
		std::string tok = text.substr(pos, comma - pos);
		pos = comma + 1;
		tok.erase(0, tok.find_first_not_of(" \t\n"));
		tok.erase(tok.find_last_not_of(" \t\n") + 1);
		if (tok.empty()) continue;
		std::smatch m;
		if (std::regex_match(tok, m, COMBO)) {
			int r1 = rank_of(m[1].str()[0]), r2 = rank_of(m[3].str()[0]);
			if (r1 == r2 && m[2] == m[4]) throw std::runtime_error("bad combo " + tok);
			add(class_of(r1, r2, m[2] == m[4]), 1);
		} else if (std::regex_match(tok, m, PAIR)) {
			int low = rank_of(m[1].str()[0]);
			for (int r = low; r <= (m[2].matched ? 12 : low); r++) add(class_of(r, r, false), 6);
		} else if (std::regex_match(tok, m, PAIR_SPAN)) {
			int a = rank_of(m[1].str()[0]), b = rank_of(m[2].str()[0]);
			for (int r = std::min(a, b); r <= std::max(a, b); r++) add(class_of(r, r, false), 6);
		} else if (std::regex_match(tok, m, HAND)) {
			int hi = rank_of(m[1].str()[0]), lo = rank_of(m[2].str()[0]);
			if (hi <= lo) throw std::runtime_error("bad token " + tok);
			int top = m[4].matched ? hi - 1 : lo;
			for (bool suited : suffixes(m[3].str()))
				for (int r = lo; r <= top; r++) add(class_of(hi, r, suited), suited ? 4 : 12);
		} else if (std::regex_match(tok, m, SPAN)) {
			int hi1 = rank_of(m[1].str()[0]), lo1 = rank_of(m[2].str()[0]);
			int hi2 = rank_of(m[4].str()[0]), lo2 = rank_of(m[5].str()[0]);
			if (m[3].str() != m[6].str()) throw std::runtime_error("bad span " + tok);
			for (bool suited : suffixes(m[3].str())) {
				if (hi1 == hi2) {
					for (int r = std::min(lo1, lo2); r <= std::max(lo1, lo2); r++) add(class_of(hi1, r, suited), suited ? 4 : 12);
				} else if (hi1 - lo1 == hi2 - lo2) {
					for (int h = std::min(hi1, hi2); h <= std::max(hi1, hi2); h++)
						add(class_of(h, h - (hi1 - lo1), suited), suited ? 4 : 12);
				} else {
					throw std::runtime_error("bad span " + tok);
				}
			}
		} else {
			throw std::runtime_error("bad token " + tok);
		}
	}
	std::vector<double> w(NCLASS);
	for (int c = 0; c < NCLASS; c++) w[c] = std::min(1.0, count[c] / combos_of_class(c));
	return w;
}

// Number of combos a range holds, rounded half to even like roundHalfEven in ranges.ts.
static int combo_count(const std::vector<double>& w) {
	double total = 0;
	for (int c = 0; c < NCLASS; c++) total += w[c] * combos_of_class(c);
	double fl = floor(total), diff = total - fl;
	if (fabs(diff - 0.5) < 1e-9) return (int)fl % 2 == 0 ? (int)fl : (int)fl + 1;
	return (int)llround(total);
}

// Sum of two ranges, capped at 1 per class (a hand can be in both lists only through different combos).
static std::vector<double> range_add(const std::vector<double>& a, const std::vector<double>& b) {
	std::vector<double> w(NCLASS);
	for (int c = 0; c < NCLASS; c++) w[c] = std::min(1.0, a[c] + b[c]);
	return w;
}
