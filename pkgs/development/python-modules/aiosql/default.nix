{ lib
, buildPythonPackage
, fetchFromGitHub
, pg8000
, pytest-asyncio
, pytestCheckHook
, pythonOlder
, setuptools
, setuptools-scm
, sphinx-rtd-theme
, sphinxHook
}:

buildPythonPackage rec {
  pname = "aiosql";
  version = "10.1";
  pyproject = true;

  disabled = pythonOlder "3.8";

  outputs = [
    "doc"
    "out"
  ];

  src = fetchFromGitHub {
    owner = "nackjicholson";
    repo = "aiosql";
    rev = "refs/tags/${version}";
    hash = "sha256-KlDwvoU0GYCN+ZCp4pp557qf9ChceS4NeA0Yiq+g3YQ=";
  };

  sphinxRoot = "docs/source";

  nativeBuildInputs = [
    setuptools
    setuptools-scm
    sphinx-rtd-theme
    sphinxHook
  ];

  propagatedBuildInputs = [
    pg8000
  ];

  nativeCheckInputs = [
    pytest-asyncio
    pytestCheckHook
  ];

  pythonImportsCheck = [
    "aiosql"
  ];

  meta = with lib; {
    description = "Simple SQL in Python";
    homepage = "https://nackjicholson.github.io/aiosql/";
    changelog = "https://github.com/nackjicholson/aiosql/releases/tag/${version}";
    license = with licenses; [ bsd2 ];
    maintainers = with maintainers; [ kaction ];
  };
}
