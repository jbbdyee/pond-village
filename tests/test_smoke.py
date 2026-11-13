"""패키지 import 확인용 스모크 테스트. 규칙 테스트가 생기면 지워도 된다."""


def test_package_imports():
    import pondvillage
    from pondvillage import rules, world  # noqa: F401
    from pondvillage.agents import base  # noqa: F401

    assert pondvillage.__doc__
