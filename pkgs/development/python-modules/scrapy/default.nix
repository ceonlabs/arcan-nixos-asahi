{ lib
, stdenv
, botocore
, buildPythonPackage
, cryptography
, cssselect
, fetchPypi
, fetchpatch
, glibcLocales
, installShellFiles
, itemadapter
, itemloaders
, jmespath
, lxml
, packaging
, parsel
, pexpect
, protego
, pydispatcher
, pyopenssl
, pytestCheckHook
, pythonOlder
, queuelib
, service-identity
, setuptools
, sybil
, testfixtures
, tldextract
, twisted
, w3lib
, zope-interface
}:

buildPythonPackage rec {
  pname = "scrapy";
  version = "2.11.1";
  pyproject = true;

  disabled = pythonOlder "3.8";

  src = fetchPypi {
    inherit version;
    pname = "Scrapy";
    hash = "sha256-czoDnHQj5StpvygQtTMgk9TkKoSEYDWcB7Auz/j3Pr4=";
  };

  patches = [
    # https://github.com/scrapy/scrapy/pull/6316
    # fix test_get_func_args. remove on next update
    (fetchpatch {
      name = "test_get_func_args.patch";
      url = "https://github.com/scrapy/scrapy/commit/b1fe97dc6c8509d58b29c61cf7801eeee1b409a9.patch";
      hash = "sha256-POlmsuW4SD9baKwZieKfmlp2vtdlb7aKQ62VOmNXsr0=";
    })
  ];

  nativeBuildInputs = [
    installShellFiles
    setuptools
  ];

  propagatedBuildInputs = [
    cryptography
    cssselect
    itemadapter
    itemloaders
    lxml
    packaging
    parsel
    protego
    pydispatcher
    pyopenssl
    queuelib
    service-identity
    tldextract
    twisted
    w3lib
    zope-interface
  ];

  nativeCheckInputs = [
    botocore
    glibcLocales
    jmespath
    pexpect
    pytestCheckHook
    sybil
    testfixtures
  ];

  LC_ALL = "en_US.UTF-8";

  disabledTestPaths = [
    "tests/test_proxy_connect.py"
    "tests/test_utils_display.py"
    "tests/test_command_check.py"
    # Don't test the documentation
    "docs"
  ];

  disabledTests = [
    # It's unclear if the failures are related to libxml2, https://github.com/NixOS/nixpkgs/pull/123890
    "test_nested_css"
    "test_nested_xpath"
    "test_flavor_detection"
    "test_follow_whitespace"
    # Requires network access
    "AnonymousFTPTestCase"
    "FTPFeedStorageTest"
    "FeedExportTest"
    "test_custom_asyncio_loop_enabled_true"
    "test_custom_loop_asyncio"
    "test_custom_loop_asyncio_deferred_signal"
    "FileFeedStoragePreFeedOptionsTest"  # https://github.com/scrapy/scrapy/issues/5157
    "test_persist"
    "test_timeout_download_from_spider_nodata_rcvd"
    "test_timeout_download_from_spider_server_hangs"
    "test_unbounded_response"
    "CookiesMiddlewareTest"
    # Depends on uvloop
    "test_asyncio_enabled_reactor_different_loop"
    "test_asyncio_enabled_reactor_same_loop"
    # Fails with AssertionError
    "test_peek_fifo"
    "test_peek_one_element"
    "test_peek_lifo"
    "test_callback_kwargs"
    # Test fails on Hydra
    "test_start_requests_laziness"
  ] ++ lib.optionals stdenv.isDarwin [
    "test_xmliter_encoding"
    "test_download"
    "test_reactor_default_twisted_reactor_select"
    "URIParamsSettingTest"
    "URIParamsFeedOptionTest"
    # flaky on darwin-aarch64
    "test_fixed_delay"
    "test_start_requests_laziness"
  ];

  postInstall = ''
    installManPage extras/scrapy.1
    installShellCompletion --cmd scrapy \
      --zsh extras/scrapy_zsh_completion \
      --bash extras/scrapy_bash_completion
  '';

  pythonImportsCheck = [
    "scrapy"
  ];

  __darwinAllowLocalNetworking = true;

  meta = with lib; {
    description = "High-level web crawling and web scraping framework";
    mainProgram = "scrapy";
    longDescription = ''
      Scrapy is a fast high-level web crawling and web scraping framework, used to crawl
      websites and extract structured data from their pages. It can be used for a wide
      range of purposes, from data mining to monitoring and automated testing.
    '';
    homepage = "https://scrapy.org/";
    changelog = "https://github.com/scrapy/scrapy/raw/${version}/docs/news.rst";
    license = licenses.bsd3;
    maintainers = with maintainers; [ vinnymeller ];
  };
}