"""Test script to check dependencies and environment for the agentic workflow."""
import sys, os
from pathlib import Path
from dotenv import load_dotenv

env_paths = [Path(__file__).parent.parent.parent.parent.parent.parent / ".env",
              Path(__file__).parent.parent.parent.parent.parent / "env" / ".env",
              Path(__file__).parent / ".env"]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path, override=False)
        break

demo_dir = Path(__file__).parent
reactxen_src = demo_dir.parent.parent.parent
if str(reactxen_src) not in sys.path:
    sys.path.insert(0, str(reactxen_src))

def check_dependencies():
    """Check if all required dependencies are installed."""
    required = {'pandas': 'pandas', 'numpy': 'numpy', 'sklearn': 'scikit-learn',
                'langchain_core': 'langchain-core', 'dotenv': 'python-dotenv'}
    print("="*70 + "\n🔍 Checking Dependencies\n" + "="*70)
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing.append(package)
    if missing:
        print(f"\n⚠️  Missing: {', '.join(missing)}\nInstall with: uv pip install {' '.join(missing)}")
        return False
    print("\n✅ All dependencies installed!")
    return True

def check_environment_variables():
    """Check if required environment variables are set."""
    required = ['WATSONX_APIKEY', 'WATSONX_PROJECT_ID', 'WATSONX_URL']
    optional = ['BRAVE_API_KEY', 'HF_BEARER_TOKEN']
    print("\n" + "="*70 + "\n🔍 Checking Environment Variables\n" + "="*70)
    missing = []
    for var in required:
        if os.environ.get(var):
            print(f"✅ {var}: {os.environ.get(var)[:10]}...")
        else:
            print(f"❌ {var}: NOT SET")
            missing.append(var)
    for var in optional:
        print(f"{'✅' if os.environ.get(var) else '⚠️ '} {var}: {'SET' if os.environ.get(var) else 'NOT SET (optional)'}")
    if missing:
        print(f"\n⚠️  Missing: {', '.join(missing)}\nSet them in your .env file or export them.")
        return False
    print("\n✅ All required environment variables are set!")
    return True

def main():
    """Main test function."""
    print("\n" + "="*70 + "\n🚀 AGENTIC WORKFLOW TEST\n" + "="*70)
    if not check_dependencies():
        print("\n❌ Please install missing dependencies: uv sync")
        sys.exit(1)
    if not check_environment_variables():
        print("\n❌ Please set required environment variables.")
        sys.exit(1)
    print("\n" + "="*70 + "\n✅ Pre-flight checks passed!\n" + "="*70)
    print("\n📝 To run the agent: python main.py")

if __name__ == "__main__":
    main()
